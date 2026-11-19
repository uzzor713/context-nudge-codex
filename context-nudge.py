#!/usr/bin/env python3
"""Remind Codex to hand off after repeated or ineffective compaction.

The hook keeps an incremental byte offset per session.  Large Codex
transcripts are append-only JSONL files, so rescanning the whole file on every
prompt is both unnecessary and likely to exceed the hook timeout.
"""

import argparse
import hashlib
import json
import mmap
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


MID_COMPACTS = 2
HIGH_COMPACTS = 3
MID_FLOOR_PERCENT = 30
HIGH_FLOOR_PERCENT = 50
MAX_INITIAL_SCAN_BYTES = 4 * 1024 * 1024
TOP_COMPACT_MARKERS = (b'"type":"compacted"', b'"type": "compacted"')
ITEM_COMPACT_MARKERS = (b'"type":"compaction"', b'"type": "compaction"')
EVENT_SCHEMA_VERSION = 1
MAX_LOG_BYTES = 10 * 1024 * 1024
LOG_RETENTION_DAYS = 90
HANDOFF_PATTERN = re.compile(
    r"交接|新(?:的)?(?:对话|会话)|另开(?:对话|会话)|handoff|hand off|new (?:chat|conversation|session|thread)",
    re.IGNORECASE,
)
URGENT_HANDOFF_PATTERN = re.compile(
    r"尽快|马上|立即|现在|当前.{0,12}(?:完成|结束).{0,12}(?:后|就)|as soon as|immediately|right now",
    re.IGNORECASE,
)


def _seek_to_record(stream, start):
    """Seek to the first complete JSONL record at or after start."""
    if start <= 0:
        stream.seek(0)
        return 0
    stream.seek(start - 1)
    previous = stream.read(1)
    stream.seek(start)
    if previous != b"\n":
        stream.readline()
    return stream.tell()


def _scan_records(path, start, stats):
    """Update stats from a bounded/incremental transcript scan."""
    with path.open("rb") as stream:
        actual_start = _seek_to_record(stream, start)
        for raw in stream:
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                stats["parse_error_count"] += 1
                continue
            if not isinstance(record, dict):
                continue

            kind = record.get("type")
            payload = record.get("payload")
            if kind == "compacted":
                stats["top_compacts"] += 1
                stats["top_floor"] = None
                stats["top_current"] = None
                continue
            if (
                kind == "response_item"
                and isinstance(payload, dict)
                and payload.get("type") == "compaction"
            ):
                stats["item_compacts"] += 1
                stats["item_floor"] = None
                stats["item_current"] = None
                continue
            if not (
                kind == "event_msg"
                and isinstance(payload, dict)
                and payload.get("type") == "token_count"
            ):
                continue

            info = payload.get("info")
            usage = info.get("last_token_usage") if isinstance(info, dict) else None
            tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
            if not isinstance(tokens, int) or tokens <= 0:
                # Codex emits a zero-valued token_count immediately after a
                # compaction. It is a bookkeeping record, not the baseline.
                continue
            size = info.get("model_context_window") if isinstance(info, dict) else None
            if isinstance(size, int) and size > 0:
                stats["window"] = size

            if stats["top_compacts"]:
                if stats["top_floor"] is None:
                    stats["top_floor"] = tokens
                stats["top_current"] = tokens
            if stats["item_compacts"]:
                if stats["item_floor"] is None:
                    stats["item_floor"] = tokens
                stats["item_current"] = tokens
        return stream.tell(), actual_start


def _marker_positions(buffer, marker):
    positions = []
    offset = 0
    while True:
        offset = buffer.find(marker, offset)
        if offset < 0:
            return positions
        positions.append(offset)
        offset += len(marker)


def _all_marker_positions(buffer, markers):
    return sorted(position for marker in markers for position in _marker_positions(buffer, marker))


def _initial_scan_start(path, size, stats):
    """Locate the last compaction without parsing the full transcript."""
    if size == 0:
        return 0
    try:
        with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            top_positions = _all_marker_positions(data, TOP_COMPACT_MARKERS)
            item_positions = _all_marker_positions(data, ITEM_COMPACT_MARKERS)
            stats["top_compacts"] = len(top_positions)
            stats["item_compacts"] = len(item_positions)
            positions = top_positions if top_positions else item_positions
            if positions:
                line_end = data.find(b"\n", positions[-1])
                return size if line_end < 0 else line_end + 1
    except (OSError, ValueError):
        pass
    return max(0, size - MAX_INITIAL_SCAN_BYTES)


def transcript_stats(path, state=None):
    """Return stats and updated scan state without rescanning old JSONL."""
    state = state if isinstance(state, dict) else {}
    try:
        size = path.stat().st_size
    except OSError:
        return {
            "compacts": 0,
            "floor": None,
            "current": None,
            "window": None,
            "state": state,
            "diagnostics": {
                "transcript_size_bytes": None,
                "scan_start_offset": None,
                "scan_end_offset": None,
                "scanned_bytes": 0,
                "initial_marker_scan": False,
                "parse_error_count": 0,
                "scan_ok": False,
            },
        }

    same_transcript = state.get("transcript_path") == str(path)
    offset = state.get("transcript_offset", 0) if same_transcript else 0
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 0

    stats = {
        "top_compacts": int(state.get("top_compacts", 0) or 0) if same_transcript else 0,
        "item_compacts": int(state.get("item_compacts", 0) or 0) if same_transcript else 0,
        "top_floor": state.get("top_floor") if same_transcript else None,
        "item_floor": state.get("item_floor") if same_transcript else None,
        "top_current": state.get("top_current") if same_transcript else None,
        "item_current": state.get("item_current") if same_transcript else None,
        "window": state.get("window") if same_transcript else None,
        "parse_error_count": 0,
    }

    initial_marker_scan = False
    if not same_transcript or offset < 0 or offset > size:
        initial_marker_scan = True
        offset = _initial_scan_start(path, size, stats)

    try:
        new_offset, actual_start = _scan_records(path, offset, stats)
    except OSError:
        return {
            "compacts": max(stats["top_compacts"], stats["item_compacts"]),
            "floor": stats["top_floor"] if stats["top_compacts"] else stats["item_floor"],
            "current": stats["top_current"] if stats["top_compacts"] else stats["item_current"],
            "window": stats["window"],
            "state": state,
            "diagnostics": {
                "transcript_size_bytes": size,
                "scan_start_offset": offset,
                "scan_end_offset": None,
                "scanned_bytes": 0,
                "initial_marker_scan": initial_marker_scan,
                "parse_error_count": stats["parse_error_count"],
                "scan_ok": False,
            },
        }

    use_top = stats["top_compacts"] > 0
    selected = "top" if use_top else "item"
    next_state = dict(state)
    next_state.update(
        {
            "transcript_path": str(path),
            "transcript_offset": new_offset,
            "top_compacts": stats["top_compacts"],
            "item_compacts": stats["item_compacts"],
            "top_floor": stats["top_floor"],
            "item_floor": stats["item_floor"],
            "top_current": stats["top_current"],
            "item_current": stats["item_current"],
            "window": stats["window"],
        }
    )
    # Keep the scan start visible for tests and diagnostics. It is not used as
    # a logical offset because the saved offset is always the EOF byte offset.
    next_state["last_scan_start"] = actual_start
    return {
        "compacts": stats[f"{selected}_compacts"],
        "floor": stats[f"{selected}_floor"],
        "current": stats[f"{selected}_current"],
        "window": stats["window"],
        "state": next_state,
        "diagnostics": {
            "transcript_size_bytes": size,
            "scan_start_offset": actual_start,
            "scan_end_offset": new_offset,
            "scanned_bytes": max(0, new_offset - actual_start),
            "initial_marker_scan": initial_marker_scan,
            "parse_error_count": stats["parse_error_count"],
            "scan_ok": True,
        },
    }


def _load_state(path):
    try:
        state = json.loads(path.read_text())
        return (state, True) if isinstance(state, dict) else ({}, False)
    except FileNotFoundError:
        return {}, True
    except (OSError, ValueError, TypeError):
        return {}, False


def _write_state(directory, path, state):
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        path.write_text(json.dumps(state, ensure_ascii=False) + "\n")
        path.chmod(0o600)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _prune_logs(directory, now):
    cutoff = now.timestamp() - LOG_RETENTION_DAYS * 24 * 60 * 60
    try:
        paths = list(directory.glob("events-*.jsonl"))
    except OSError:
        return
    for path in paths:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _event_log_path(directory, timestamp):
    base = directory / f"events-{timestamp:%Y-%m}.jsonl"
    path = base
    sequence = 0
    while True:
        try:
            if not path.exists() or path.stat().st_size < MAX_LOG_BYTES:
                return path
        except OSError:
            return path
        sequence += 1
        path = directory / f"events-{timestamp:%Y-%m}.{sequence}.jsonl"


def _append_event(directory, event):
    """Append one privacy-safe event without making the hook blocking."""
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        timestamp = datetime.now(timezone.utc)
        _prune_logs(directory, timestamp)
        path = _event_log_path(directory, timestamp)
        payload = dict(event)
        payload.setdefault("schema_version", EVENT_SCHEMA_VERSION)
        payload.setdefault("timestamp", timestamp.isoformat())
        line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _usage_fields(compacts, event_compacts, transcript_compacts, floor, current, window):
    floor_percent = floor * 100 // window if floor is not None and window else None
    current_percent = current * 100 // window if current is not None and window else None
    return {
        "compact_count": compacts,
        "event_compact_count": event_compacts,
        "transcript_compact_count": transcript_compacts,
        "floor_tokens": floor,
        "floor_percent": floor_percent,
        "current_tokens": current,
        "current_percent": current_percent,
        "context_window": window,
    }


def _message_text(payload):
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    )


def _hook_reminder(payload):
    if payload.get("type") != "message" or payload.get("role") != "developer":
        return None
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    if not isinstance(metadata, dict):
        return None
    content_item_kinds = metadata.get("content_item_kinds")
    if not isinstance(content_item_kinds, list) or "hooks.additional_context" not in content_item_kinds:
        return None
    item_metadata = metadata.get("content_item_metadata")
    if not isinstance(item_metadata, list) or not any(
        isinstance(item, dict)
        and isinstance(item.get("provenance"), dict)
        and item["provenance"].get("hook") is True
        for item in item_metadata
    ):
        return None
    text = _message_text(payload)
    if not text.startswith("[context-nudge]"):
        return None
    return {
        "id": str(payload.get("id") or text),
        "level": 2 if "尽快" in text else 1,
    }


def _compliance_result(session_hash, pending, final_text=None):
    mentioned = bool(final_text and HANDOFF_PATTERN.search(final_text))
    followed = mentioned and (
        pending["level"] == 1 or bool(URGENT_HANDOFF_PATTERN.search(final_text or ""))
    )
    return {
        "event": "agent_compliance",
        "session_hash": session_hash,
        "reminder_hash": hashlib.sha256(pending["id"].encode()).hexdigest(),
        "level": pending["level"],
        "agent_context_observed": True,
        "final_answer_observed": final_text is not None,
        "agent_mentioned_handoff": mentioned,
        "agent_followed_level": followed,
    }


def analyze_transcript(path):
    """Return privacy-safe compliance results for hook reminders in one transcript."""
    session_hash = None
    pending = None
    results = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        return results
    with stream:
        for raw in stream:
            try:
                record = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "session_meta" and session_hash is None:
                session_id = str(payload.get("session_id") or payload.get("id") or "")
                if session_id:
                    session_hash = hashlib.sha256(session_id.encode()).hexdigest()
                continue
            if record.get("type") != "response_item":
                continue

            reminder = _hook_reminder(payload)
            if reminder is not None:
                if pending is not None and session_hash is not None:
                    results.append(_compliance_result(session_hash, pending))
                pending = reminder
                continue
            if pending is None or session_hash is None or payload.get("type") != "message":
                continue
            if payload.get("role") == "assistant" and payload.get("phase") == "final_answer":
                results.append(_compliance_result(session_hash, pending, _message_text(payload)))
                pending = None
            elif payload.get("role") == "user":
                results.append(_compliance_result(session_hash, pending))
                pending = None
    if pending is not None and session_hash is not None:
        results.append(_compliance_result(session_hash, pending))
    return results


def _existing_compliance_keys(log_dir):
    keys = set()
    try:
        paths = sorted(log_dir.glob("events-*.jsonl"))
    except OSError:
        return keys
    for path in paths:
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    continue
                if isinstance(event, dict) and event.get("event") == "agent_compliance":
                    keys.add((event.get("session_hash"), event.get("reminder_hash")))
    return keys


def _read_log_events(log_dir):
    events = []
    invalid_lines = 0
    try:
        paths = sorted(log_dir.glob("events-*.jsonl"))
    except OSError:
        return events, invalid_lines
    for path in paths:
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError:
            continue
        with stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    invalid_lines += 1
                    continue
                if isinstance(event, dict):
                    events.append(event)
                else:
                    invalid_lines += 1
    return events, invalid_lines


def _numeric_summary(values):
    numbers = sorted(value for value in values if isinstance(value, (int, float)))
    if not numbers:
        return {"count": 0, "p50": None, "p95": None, "max": None}

    def percentile(fraction):
        index = max(0, int((len(numbers) * fraction) + 0.999999) - 1)
        return numbers[index]

    return {
        "count": len(numbers),
        "p50": statistics.median(numbers),
        "p95": percentile(0.95),
        "max": numbers[-1],
    }


def build_stats(events, invalid_lines=0):
    sessions = set()
    reminder_levels = Counter()
    trigger_reasons = Counter()
    suppression_reasons = Counter()
    session_compacts = {}
    floor_percents = []
    current_percents = []
    started = set()
    finished = set()
    hook_durations = []
    compliance_total = 0
    compliance_final = 0
    compliance_mentioned = 0
    compliance_followed = 0

    for event in events:
        session_hash = event.get("session_hash")
        if isinstance(session_hash, str) and session_hash:
            sessions.add(session_hash)
        event_name = event.get("event")
        if event_name == "evaluation":
            compacts = event.get("compact_count")
            if isinstance(session_hash, str) and isinstance(compacts, int):
                session_compacts[session_hash] = max(session_compacts.get(session_hash, 0), compacts)
            if event.get("decision") == "suppressed" and event.get("suppressed_reason"):
                suppression_reasons[str(event["suppressed_reason"])] += 1
            floor_percents.append(event.get("floor_percent"))
            current_percents.append(event.get("current_percent"))
        elif event_name == "reminder_emitted":
            level = event.get("level")
            if isinstance(level, int):
                reminder_levels[str(level)] += 1
            reasons = event.get("reason")
            if isinstance(reasons, list):
                trigger_reasons.update(str(reason) for reason in reasons)
        elif event_name == "hook_started":
            invocation_id = event.get("invocation_id")
            if invocation_id:
                started.add(invocation_id)
        elif event_name == "hook_finished":
            invocation_id = event.get("invocation_id")
            if invocation_id:
                finished.add(invocation_id)
            hook_durations.append(event.get("hook_duration_ms"))
        elif event_name == "agent_compliance":
            compliance_total += 1
            compliance_final += int(event.get("final_answer_observed") is True)
            compliance_mentioned += int(event.get("agent_mentioned_handoff") is True)
            compliance_followed += int(event.get("agent_followed_level") is True)

    compact_distribution = Counter(session_compacts.values())
    return {
        "events_read": len(events),
        "invalid_log_lines": invalid_lines,
        "sessions_observed": len(sessions),
        "compact_total_across_sessions": sum(session_compacts.values()),
        "compact_distribution": {
            str(key): compact_distribution[key] for key in sorted(compact_distribution)
        },
        "reminders_total": sum(reminder_levels.values()),
        "reminders_by_level": dict(sorted(reminder_levels.items())),
        "trigger_reasons": dict(sorted(trigger_reasons.items())),
        "suppression_reasons": dict(sorted(suppression_reasons.items())),
        "floor_percent": _numeric_summary(floor_percents),
        "current_percent": _numeric_summary(current_percents),
        "hook_duration_ms": _numeric_summary(hook_durations),
        "started_without_finished": len(started - finished),
        "agent_compliance": {
            "total": compliance_total,
            "final_answer_observed": compliance_final,
            "mentioned_handoff": compliance_mentioned,
            "followed_level": compliance_followed,
            "mentioned_rate": (
                round(compliance_mentioned / compliance_total, 4) if compliance_total else None
            ),
            "followed_rate": (
                round(compliance_followed / compliance_total, 4) if compliance_total else None
            ),
        },
    }


def stats_command(codex_home):
    events, invalid_lines = _read_log_events(codex_home / "context-nudge-logs")
    print(json.dumps(build_stats(events, invalid_lines), ensure_ascii=False, indent=2))


def analyze_command(paths, codex_home):
    log_dir = codex_home / "context-nudge-logs"
    existing = _existing_compliance_keys(log_dir)
    transcript_paths = []
    for path in paths:
        if path.is_dir():
            transcript_paths.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            transcript_paths.append(path)

    summary = {
        "transcripts_scanned": len(transcript_paths),
        "reminders_found": 0,
        "events_written": 0,
        "agent_mentioned_handoff": 0,
        "agent_followed_level": 0,
    }
    for path in transcript_paths:
        for result in analyze_transcript(path):
            summary["reminders_found"] += 1
            summary["agent_mentioned_handoff"] += int(result["agent_mentioned_handoff"])
            summary["agent_followed_level"] += int(result["agent_followed_level"])
            key = (result["session_hash"], result["reminder_hash"])
            if key not in existing and _append_event(log_dir, result):
                existing.add(key)
                summary["events_written"] += 1
    print(json.dumps(summary, ensure_ascii=False))


def cli():
    parser = argparse.ArgumentParser(description="Codex context handoff reminder")
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="analyze Agent handling of hook reminders")
    analyze.add_argument("paths", nargs="+", type=Path, help="transcript files or directories")
    analyze.add_argument("--codex-home", type=Path)
    stats = subparsers.add_parser("stats", help="summarize local context-nudge events")
    stats.add_argument("--codex-home", type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        codex_home = args.codex_home or Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        analyze_command(args.paths, codex_home)
    elif args.command == "stats":
        codex_home = args.codex_home or Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        stats_command(codex_home)
    else:
        main()


def main():
    started = time.monotonic()
    log_dir = None
    lifecycle = None
    lifecycle_details = {"status": "finished"}
    try:
        event = json.load(sys.stdin)
        if not isinstance(event, dict) or event.get("hook_event_name") not in (
            "UserPromptSubmit",
            "SessionStart",
        ):
            return
        session_id = str(event.get("session_id") or "")
        if not session_id:
            return

        codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        state_dir = codex_home / "context-nudge-state"
        log_dir = codex_home / "context-nudge-logs"
        session_hash = hashlib.sha256(session_id.encode()).hexdigest()
        lifecycle = {
            "invocation_id": uuid.uuid4().hex,
            "hook_event_name": event["hook_event_name"],
            "session_hash": session_hash,
        }
        _append_event(log_dir, {"event": "hook_started", **lifecycle})
        state_path = state_dir / (session_hash + ".json")
        state, state_read_ok = _load_state(state_path)
        lifecycle_details["state_read_ok"] = state_read_ok

        if event.get("hook_event_name") == "SessionStart":
            state_write_ok = True
            if event.get("source") == "compact":
                state["event_compacts"] = int(state.get("event_compacts", 0) or 0) + 1
                state_write_ok = _write_state(state_dir, state_path, state)
            lifecycle_details["state_write_ok"] = state_write_ok
            return

        prompt = str(event.get("prompt") or "")
        transcript = event.get("transcript_path")
        transcript_available = False
        diagnostics = {
            "transcript_size_bytes": None,
            "scan_start_offset": None,
            "scan_end_offset": None,
            "scanned_bytes": 0,
            "initial_marker_scan": False,
            "parse_error_count": 0,
            "scan_ok": False,
        }
        stats = {
            "compacts": 0,
            "floor": None,
            "current": None,
            "window": None,
            "state": state,
            "diagnostics": diagnostics,
        }
        if isinstance(transcript, str) and transcript.endswith(".jsonl"):
            transcript_path = Path(transcript)
            if transcript_path.is_file():
                transcript_available = True
                stats = transcript_stats(transcript_path, state)
                diagnostics = stats["diagnostics"]
        lifecycle_details.update(diagnostics)

        state = stats["state"]
        transcript_compacts = stats["compacts"]
        event_compacts = int(state.get("event_compacts", 0) or 0)
        compacts = max(transcript_compacts, event_compacts)
        floor = stats["floor"]
        current = stats["current"]
        window = stats["window"]
        usage_fields = _usage_fields(
            compacts, event_compacts, transcript_compacts, floor, current, window
        )
        floor_percent = usage_fields["floor_percent"] or 0
        current_percent = usage_fields["current_percent"] or 0
        state.update({"compacts": compacts, "floor_tokens": floor, "current_tokens": current})
        state_write_ok = _write_state(state_dir, state_path, state)
        lifecycle_details["state_write_ok"] = state_write_ok

        if not compacts:
            level = 0
        else:
            level = 0
            if compacts >= MID_COMPACTS or floor_percent >= MID_FLOOR_PERCENT:
                level = 1
            if compacts >= HIGH_COMPACTS or floor_percent >= HIGH_FLOOR_PERCENT:
                level = 2

        reminded = int(state.get("reminded_level", 0) or 0)
        suppressed_reason = None
        if prompt.startswith("/"):
            suppressed_reason = "slash_command"
        elif not state_read_ok or not state_write_ok:
            suppressed_reason = "state_error"
        elif not level:
            if not transcript_available:
                suppressed_reason = "missing_transcript"
            elif diagnostics["parse_error_count"]:
                suppressed_reason = "parse_error"
            elif not compacts:
                suppressed_reason = "no_compaction"
            elif floor is None or current is None or not window:
                suppressed_reason = "missing_token_usage"
            else:
                suppressed_reason = "below_threshold"
        elif level <= reminded:
            suppressed_reason = "already_reminded"

        _append_event(
            codex_home / "context-nudge-logs",
            {
                "event": "evaluation",
                "session_hash": session_hash,
                "computed_level": level,
                "previous_reminded_level": reminded,
                "decision": "suppressed" if suppressed_reason else "emitted",
                "suppressed_reason": suppressed_reason,
                "state_read_ok": state_read_ok,
                "state_write_ok": state_write_ok,
                **diagnostics,
                **usage_fields,
            },
        )
        lifecycle_details.update(
            {
                "decision": "suppressed" if suppressed_reason else "emitted",
                "suppressed_reason": suppressed_reason,
            }
        )
        if suppressed_reason:
            return

        previous_reminded_level = reminded
        state["reminded_level"] = level
        state["last_reminder_at"] = int(time.time())
        reminder_state_write_ok = _write_state(state_dir, state_path, state)
        lifecycle_details["state_write_ok"] = reminder_state_write_ok

        facts = f"本会话已压缩 {compacts} 次"
        if floor is not None and window:
            facts += f"，压缩后基线占窗口 {floor_percent}%（当前约 {current_percent}%）"
        urgency = "建议在合适节点" if level == 1 else "建议完成当前这一小步后尽快"
        reminder = f"[context-nudge] {facts}。先完成用户当前请求，在回答末尾用一句话{urgency}生成交接摘要并开新对话；不要打断当前任务或单独回复。"
        notice = f"Context 已压缩 {compacts} 次，可考虑交接到新对话。"
        reasons = []
        if compacts >= (HIGH_COMPACTS if level == 2 else MID_COMPACTS):
            reasons.append("compact_count")
        if floor_percent >= (HIGH_FLOOR_PERCENT if level == 2 else MID_FLOOR_PERCENT):
            reasons.append("floor_percent")
        _append_event(
            log_dir,
            {
                "event": "reminder_emitted",
                "session_hash": session_hash,
                "level": level,
                "reason": reasons,
                **usage_fields,
                "previous_reminded_level": previous_reminded_level,
                "system_message_emitted": True,
                "additional_context_emitted": True,
                "state_write_ok": reminder_state_write_ok,
                "hook_duration_ms": round((time.monotonic() - started) * 1000, 3),
            },
        )
        print(
            json.dumps(
                {
                    "systemMessage": notice,
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": reminder,
                    },
                },
                ensure_ascii=False,
            )
        )
    except Exception:
        # Advisory hook: malformed or unavailable input must not block a prompt.
        lifecycle_details["status"] = "error"
        return
    finally:
        if log_dir is not None and lifecycle is not None:
            _append_event(
                log_dir,
                {
                    "event": "hook_finished",
                    **lifecycle,
                    **lifecycle_details,
                    "hook_duration_ms": round((time.monotonic() - started) * 1000, 3),
                },
            )


if __name__ == "__main__":
    cli()
