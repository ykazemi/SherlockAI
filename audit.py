# audit.py
from __future__ import annotations
from typing import Dict, Any
import json
from pathlib import Path
from datetime import datetime

AUDIT_PATH = Path("audit_log.jsonl")


def append_audit(event: Dict[str, Any]) -> None:
    event = dict(event)
    event["logged_at"] = datetime.now().isoformat()
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
