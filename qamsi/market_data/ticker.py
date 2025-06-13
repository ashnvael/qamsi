from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ticker:
    name: str
    code: str
    currency: str | None = None
