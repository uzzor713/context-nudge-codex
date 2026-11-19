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
            env = dict(os.environ, CODEX_HOME=str(tmp_path / "codex"))
            event = {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "test-session",
                "prompt": "继续",
                "transcript_path": str(transcript),
            }
            output = subprocess.check_output(
                [sys.executable, str(ROOT / "context-nudge.py")],
                input=(json.dumps(event) + "\n").encode(),
                env=env,
            )
            result = json.loads(output)
            self.assertIn("systemMessage", result)
            self.assertIn("additionalContext", result["hookSpecificOutput"])
            self.assertIn("压缩后基线占窗口 7%", result["hookSpecificOutput"]["additionalContext"])

            log_files = list((tmp_path / "codex" / "context-nudge-logs").glob("events-*.jsonl"))
            self.assertEqual(len(log_files), 1)
            event_log = json.loads(log_files[0].read_text())
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
            self.assertEqual(log_files[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(log_files[0].parent.stat().st_mode & 0o777, 0o700)

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


if __name__ == "__main__":
    unittest.main()
