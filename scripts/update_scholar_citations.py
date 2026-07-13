#!/usr/bin/env python3
"""Refresh the cached Google Scholar citation count for the static site."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "assets" / "scholar-citations.json"
PROFILE_URL = "https://scholar.google.com/citations?user=TQIKwMIAAAAJ&hl=en"


def fetch_profile() -> str:
    request = Request(
        PROFILE_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_citations(html: str) -> int | None:
    patterns = (
        r'<td class="gsc_rsb_sc1"><a[^>]*>Citations</a></td><td class="gsc_rsb_std">([\d,]+)</td>',
        r"Cited by\s*([\d,]+)",
    )

    for pattern in patterns:
        match = re.search(pattern, html, re.S)
        if match:
            return int(match.group(1).replace(",", ""))

    return None


def main() -> int:
    try:
        html = fetch_profile()
        citations = parse_citations(html)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Could not fetch Google Scholar profile: {exc}", file=sys.stderr)
        return 0

    if not citations:
        print("Could not parse citation count; keeping cached value.", file=sys.stderr)
        return 0

    payload = {
        "citations": citations,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Google Scholar",
        "profile": PROFILE_URL,
    }
    DATA_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Updated Google Scholar citations: {citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
