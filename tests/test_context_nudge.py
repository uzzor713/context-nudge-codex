import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("context_nudge", ROOT / "context-nudge.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def compacted():
    return {"type": "compacted", "payload": {}}


def token_count(tokens):
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {"input_tokens": tokens},
                "model_context_window": 258400,
            },
        },
    }


class ContextNudgeTests(unittest.TestCase):
    def run_hook(self, codex_home, event):
        env = dict(os.environ, CODEX_HOME=str(codex_home))
        return subprocess.check_output(
            [sys.executable, str(ROOT / "context-nudge.py")],
            input=(json.dumps(event) + "\n").encode(),
            env=env,
        )

    def test_zero_token_count_is_not_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [compacted(), token_count(0), token_count(36786), token_count(46899)]
                )
                + "\n"
            )
            result = MODULE.transcript_stats(transcript, {})
            self.assertEqual(result["compacts"], 1)
            self.assertEqual(result["floor"], 36786)
            self.assertEqual(result["current"], 46899)
            self.assertGreater(result["state"]["last_scan_start"], -1)
            self.assertTrue(result["diagnostics"]["initial_marker_scan"])
            self.assertTrue(result["diagnostics"]["scan_ok"])
            self.assertEqual(result["diagnostics"]["parse_error_count"], 0)
            self.assertGreater(result["diagnostics"]["scanned_bytes"], 0)

    def test_incremental_scan_only_reads_appended_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text(json.dumps(compacted()) + "\n" + json.dumps(token_count(10000)) + "\n")
            first = MODULE.transcript_stats(transcript, {})
            old_offset = first["state"]["transcript_offset"]
            with transcript.open("a") as stream:
                stream.write(json.dumps(token_count(20000)) + "\n")
            second = MODULE.transcript_stats(transcript, first["state"])
            self.assertEqual(second["floor"], 10000)
            self.assertEqual(second["current"], 20000)
            self.assertEqual(second["state"]["last_scan_start"], old_offset)

    def test_hook_emits_agent_context_after_two_compacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            transcript = tmp_path / "session.jsonl"
            transcript.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [
                        compacted(),
                        token_count(10000),
                        compacted(),
                        token_count(20000),
                    ]
                )
                + "\n"
            )
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "test-session",
                "prompt": "继续",
                "transcript_path": str(transcript),
            }
            output = self.run_hook(tmp_path / "codex", event)
            result = json.loads(output)
            self.assertIn("systemMessage", result)
            self.assertIn("additionalContext", result["hookSpecificOutput"])
            self.assertIn("压缩后基线占窗口 7%", result["hookSpecificOutput"]["additionalContext"])

            log_files = list((tmp_path / "codex" / "context-nudge-logs").glob("events-*.jsonl"))
            self.assertEqual(len(log_files), 1)
            logged_events = [json.loads(line) for line in log_files[0].read_text().splitlines()]
            event_log = next(event for event in logged_events if event["event"] == "reminder_emitted")
            started = next(event for event in logged_events if event["event"] == "hook_started")
            finished = next(event for event in logged_events if event["event"] == "hook_finished")
            self.assertEqual(event_log["schema_version"], 1)
            self.assertEqual(event_log["event"], "reminder_emitted")
            self.assertEqual(event_log["session_hash"], MODULE.hashlib.sha256(b"test-session").hexdigest())
            self.assertEqual(event_log["level"], 1)
            self.assertEqual(event_log["reason"], ["compact_count"])
            self.assertEqual(event_log["transcript_compact_count"], 2)
            self.assertEqual(event_log["floor_tokens"], 20000)
            self.assertEqual(event_log["floor_percent"], 7)
            self.assertTrue(event_log["system_message_emitted"])
            self.assertTrue(event_log["additional_context_emitted"])
            self.assertTrue(event_log["state_write_ok"])
            self.assertEqual(started["invocation_id"], finished["invocation_id"])
            self.assertEqual(finished["status"], "finished")
            self.assertTrue(finished["state_read_ok"])
            self.assertTrue(finished["state_write_ok"])
            self.assertTrue(finished["initial_marker_scan"])
            self.assertGreater(finished["scanned_bytes"], 0)
            self.assertGreaterEqual(finished["hook_duration_ms"], 0)
            self.assertEqual(log_files[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(log_files[0].parent.stat().st_mode & 0o777, 0o700)

    def test_hook_logs_suppressed_evaluation_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            codex_home = tmp_path / "codex"
            empty_transcript = tmp_path / "empty.jsonl"
            empty_transcript.write_text(json.dumps(token_count(10000)) + "\n")
            compacted_transcript = tmp_path / "compacted.jsonl"
            compacted_transcript.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [compacted(), token_count(10000), compacted(), token_count(20000)]
                )
                + "\n"
            )
            base_event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "evaluation-session",
                "prompt": "继续",
            }

            self.assertEqual(
                self.run_hook(codex_home, {**base_event, "transcript_path": str(empty_transcript)}),
                b"",
            )
            self.assertEqual(
                self.run_hook(
                    codex_home,
                    {**base_event, "prompt": "/status", "transcript_path": str(compacted_transcript)},
                ),
                b"",
            )
            self.assertTrue(
                self.run_hook(codex_home, {**base_event, "transcript_path": str(compacted_transcript)})
            )
            self.assertEqual(
                self.run_hook(codex_home, {**base_event, "transcript_path": str(compacted_transcript)}),
                b"",
            )

            log_file = next((codex_home / "context-nudge-logs").glob("events-*.jsonl"))
            events = [json.loads(line) for line in log_file.read_text().splitlines()]
            evaluations = [event for event in events if event["event"] == "evaluation"]
            self.assertEqual(
                [event["suppressed_reason"] for event in evaluations],
                ["no_compaction", "slash_command", None, "already_reminded"],
            )
            self.assertEqual(
                [event["decision"] for event in evaluations],
                ["suppressed", "suppressed", "emitted", "suppressed"],
            )

    def test_hook_logs_parse_and_state_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            malformed = tmp_path / "malformed.jsonl"
            malformed.write_text("not-json\n")
            parse_home = tmp_path / "parse-home"
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "diagnostic-session",
                "prompt": "继续",
                "transcript_path": str(malformed),
            }
            self.assertEqual(self.run_hook(parse_home, event), b"")
            parse_log = next((parse_home / "context-nudge-logs").glob("events-*.jsonl"))
            parse_events = [json.loads(line) for line in parse_log.read_text().splitlines()]
            parse_evaluation = next(item for item in parse_events if item["event"] == "evaluation")
            self.assertEqual(parse_evaluation["suppressed_reason"], "parse_error")
            self.assertEqual(parse_evaluation["parse_error_count"], 1)

            state_home = tmp_path / "state-home"
            state_home.mkdir()
            (state_home / "context-nudge-state").write_text("blocks-state-directory")
            transcript = tmp_path / "state-error.jsonl"
            transcript.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in [compacted(), token_count(10000), compacted(), token_count(20000)]
                )
                + "\n"
            )
            event["transcript_path"] = str(transcript)
            self.assertEqual(self.run_hook(state_home, event), b"")
            state_log = next((state_home / "context-nudge-logs").glob("events-*.jsonl"))
            state_events = [json.loads(line) for line in state_log.read_text().splitlines()]
            state_evaluation = next(item for item in state_events if item["event"] == "evaluation")
            self.assertEqual(state_evaluation["suppressed_reason"], "state_error")
            self.assertFalse(state_evaluation["state_read_ok"])
            self.assertFalse(state_evaluation["state_write_ok"])

    def test_initial_scan_counts_compacts_before_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            filler = {"type": "response_item", "payload": {"text": "x" * 1024}}
            records = [compacted(), token_count(10000)]
            records.extend(filler for _ in range(5000))
            records.extend([compacted(), token_count(30000)])
            transcript.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            result = MODULE.transcript_stats(transcript, {})
            self.assertEqual(result["compacts"], 2)
            self.assertEqual(result["floor"], 30000)
            self.assertGreater(result["state"]["last_scan_start"], 4 * 1024 * 1024)

    def test_offline_analysis_logs_agent_compliance_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            transcript = tmp_path / "analysis.jsonl"
            records = [
                {
                    "type": "session_meta",
                    "payload": {"id": "analysis-session", "session_id": "analysis-session"},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "id": "reminder-message",
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "[context-nudge] 本会话已压缩 3 次。建议完成当前这一小步后尽快生成交接摘要并开新对话。",
                            }
                        ],
                        "internal_chat_message_metadata_passthrough": {
                            "content_item_kinds": ["hooks.additional_context"],
                            "content_item_metadata": [{"provenance": {"hook": True}}],
                        },
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "当前步骤已完成。请尽快生成交接摘要并开新对话。",
                            }
                        ],
                    },
                },
            ]
            transcript.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            codex_home = tmp_path / "codex"
            command = [
                sys.executable,
                str(ROOT / "context-nudge.py"),
                "analyze",
                str(transcript),
                "--codex-home",
                str(codex_home),
            ]
            first = json.loads(subprocess.check_output(command))
            second = json.loads(subprocess.check_output(command))
            self.assertEqual(first["reminders_found"], 1)
            self.assertEqual(first["events_written"], 1)
            self.assertEqual(second["events_written"], 0)

            log_file = next((codex_home / "context-nudge-logs").glob("events-*.jsonl"))
            events = [json.loads(line) for line in log_file.read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "agent_compliance")
            self.assertEqual(events[0]["level"], 2)
            self.assertTrue(events[0]["agent_context_observed"])
            self.assertTrue(events[0]["final_answer_observed"])
            self.assertTrue(events[0]["agent_mentioned_handoff"])
            self.assertTrue(events[0]["agent_followed_level"])
            self.assertNotIn("当前步骤已完成", log_file.read_text())


if __name__ == "__main__":
    unittest.main()
