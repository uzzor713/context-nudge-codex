#!/usr/bin/env python3
"""Remind Codex to hand off after repeated or ineffective compaction.

The hook keeps an incremental byte offset per session.  Large Codex
transcripts are append-only JSONL files, so rescanning the whole file on every
prompt is both unnecessary and likely to exceed the hook timeout.
"""

import hashlib
import json
import mmap
import os
import sys
import time
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
    }

    if not same_transcript or offset < 0 or offset > size:
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
    }


def _load_state(path):
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_state(directory, path, state):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False) + "\n")
    path.chmod(0o600)


def _append_event(directory, event):
    """Append one privacy-safe event without making the hook blocking."""
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        timestamp = datetime.now(timezone.utc)
        path = directory / f"events-{timestamp:%Y-%m}.jsonl"
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


def main():
    started = time.monotonic()
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
        session_hash = hashlib.sha256(session_id.encode()).hexdigest()
        state_path = state_dir / (session_hash + ".json")
        state = _load_state(state_path)

        if event.get("hook_event_name") == "SessionStart":
            if event.get("source") == "compact":
                state["event_compacts"] = int(state.get("event_compacts", 0) or 0) + 1
                _write_state(state_dir, state_path, state)
            return

        prompt = str(event.get("prompt") or "")
        transcript = event.get("transcript_path")
        stats = {"compacts": 0, "floor": None, "current": None, "window": None, "state": state}
        if isinstance(transcript, str) and transcript.endswith(".jsonl"):
            transcript_path = Path(transcript)
            if transcript_path.is_file():
                stats = transcript_stats(transcript_path, state)

        state = stats["state"]
        transcript_compacts = stats["compacts"]
        event_compacts = int(state.get("event_compacts", 0) or 0)
        compacts = max(transcript_compacts, event_compacts)
        floor = stats["floor"]
        current = stats["current"]
        window = stats["window"]
        state.update({"compacts": compacts, "floor_tokens": floor, "current_tokens": current})
        _write_state(state_dir, state_path, state)

        if not compacts:
            return
        floor_percent = floor * 100 // window if floor is not None and window else 0
        current_percent = current * 100 // window if current is not None and window else 0
        level = 0
        if compacts >= MID_COMPACTS or floor_percent >= MID_FLOOR_PERCENT:
            level = 1
        if compacts >= HIGH_COMPACTS or floor_percent >= HIGH_FLOOR_PERCENT:
            level = 2
        if not level or prompt.startswith("/"):
            return

        reminded = int(state.get("reminded_level", 0) or 0)
        if level <= reminded:
            return
        previous_reminded_level = reminded
        state["reminded_level"] = level
        state["last_reminder_at"] = int(time.time())
        _write_state(state_dir, state_path, state)

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
            codex_home / "context-nudge-logs",
            {
                "event": "reminder_emitted",
                "session_hash": session_hash,
                "level": level,
                "reason": reasons,
                "compact_count": compacts,
                "event_compact_count": event_compacts,
                "transcript_compact_count": transcript_compacts,
                "floor_tokens": floor,
                "floor_percent": floor_percent if floor is not None and window else None,
                "current_tokens": current,
                "current_percent": current_percent if current is not None and window else None,
                "context_window": window,
                "previous_reminded_level": previous_reminded_level,
                "system_message_emitted": True,
                "additional_context_emitted": True,
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
        return


if __name__ == "__main__":
    main()
