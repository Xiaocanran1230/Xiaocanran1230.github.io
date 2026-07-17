#!/usr/bin/env python3
"""Refresh the cached Google Scholar citation count for the static site."""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "assets" / "scholar-citations.json"
INDEX_PATH = ROOT / "index.html"
PROFILE_URL = "https://scholar.google.com/citations?user=TQIKwMIAAAAJ&hl=en"


def fetch_profile() -> str:
    errors: list[str] = []

    for attempt in range(3):
        request = Request(
            f"{PROFILE_URL}&oi=ao",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(str(exc))
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError("; ".join(errors))


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


def update_index_badge(citations: int) -> bool:
    html = INDEX_PATH.read_text(encoding="utf-8")
    formatted = f"{citations:,}"
    shield_value = formatted.replace(",", "%2C")
    updated = re.sub(
        r'aria-label="[\d,]+ citations on Google Scholar"',
        f'aria-label="{formatted} citations on Google Scholar"',
        html,
    )
    updated = re.sub(
        r"citations-[\dA-Fa-f,%]+-4285F4",
        f"citations-{shield_value}-4285F4",
        updated,
    )
    updated = re.sub(
        r'alt="[\d,]+ citations"',
        f'alt="{formatted} citations"',
        updated,
    )

    if updated == html:
        return False

    INDEX_PATH.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    try:
        html = fetch_profile()
        citations = parse_citations(html)
    except RuntimeError as exc:
        print(f"Could not fetch Google Scholar profile: {exc}", file=sys.stderr)
        return 1

    if not citations:
        print("Could not parse citation count; keeping cached value.", file=sys.stderr)
        return 1

    try:
        current = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        current = {}

    data_changed = current.get("citations") != citations
    badge_changed = update_index_badge(citations)

    if data_changed:
        payload = {
            "citations": citations,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "Google Scholar",
            "profile": PROFILE_URL,
        }
        DATA_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if data_changed or badge_changed:
        print(f"Updated Google Scholar citations: {citations}")
    else:
        print(f"Google Scholar citations unchanged: {citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
