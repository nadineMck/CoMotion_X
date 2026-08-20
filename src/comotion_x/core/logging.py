"""Small structured-event logger for reproducible runs."""

from __future__ import annotations

import json
from typing import Any, TextIO


def emit_event(event: str, *, stream: TextIO | None = None, **fields: Any) -> None:
    payload = {"event": event, **fields}
    rendered = json.dumps(payload, sort_keys=True, allow_nan=False)
    if stream is None:
        print(rendered, flush=True)
    else:
        print(rendered, file=stream, flush=True)

