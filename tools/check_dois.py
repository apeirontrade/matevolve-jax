#!/usr/bin/env python3
"""
Checks every DOI cited in engine.get_universal_catalog() against the public
Crossref and DataCite REST APIs and prints the resolved metadata.
"""

import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple
import urllib.error
import urllib.parse
import urllib.request

# Ensure repo root is on sys.path when invoked as `python3 tools/check_dois.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402

DOI_PATTERN = re.compile(r"DOI:(10\.\d{4,9}/[^\s,;]+)", re.IGNORECASE)


def extract_catalog_dois() -> List[Tuple[str, str, str]]:
    """
    Returns a list of (compound_name, field_name, doi) tuples for every DOI
    present in get_universal_catalog().
    """
    records: List[Tuple[str, str, str]] = []
    for entry in engine.get_universal_catalog():
        for match in DOI_PATTERN.finditer(entry.source):
            records.append((entry.name, "source", match.group(1)))
        for prop_key, prop_cite in entry.property_sources.items():
            for match in DOI_PATTERN.finditer(prop_cite):
                records.append((entry.name, prop_key, match.group(1)))
    return records


def resolve_doi_metadata(doi: str, timeout: float = 15.0) -> Dict[str, str]:
    """
    Queries Crossref first, then DataCite, returning a dict with keys:
      provider, doi, year, first_author, title
    Raises urllib.error.URLError on network failure or ValueError if unregistered.
    """
    headers = {"User-Agent": "MatEvolve-JAX-DOIChecker/0.1"}

    # 1. Try Crossref
    cr_url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(cr_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            msg = json.loads(resp.read().decode("utf-8"))["message"]
            titles = msg.get("title") or ["(untitled)"]
            authors = msg.get("author") or []
            first_author = authors[0].get("family", "Unknown") if authors else "Unknown"
            year_parts = msg.get("issued", {}).get("date-parts", [[None]])
            year = str(year_parts[0][0]) if year_parts and year_parts[0] else "?"
            return {
                "provider": "Crossref",
                "doi": doi,
                "year": year,
                "first_author": first_author,
                "title": re.sub(r"\s+", " ", titles[0]).strip(),
            }
    except urllib.error.HTTPError as e:
        code = e.code
        e.close()
        if code != 404:
            raise

    # 2. Try DataCite
    dc_url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    req_dc = urllib.request.Request(dc_url, headers=headers)
    try:
        with urllib.request.urlopen(req_dc, timeout=timeout) as resp:
            attrs = json.loads(resp.read().decode("utf-8"))["data"]["attributes"]
            titles = attrs.get("titles") or [{}]
            title = titles[0].get("title", "(untitled)") if titles else "(untitled)"
            creators = attrs.get("creators") or []
            first_author = creators[0].get("name", "DataCite") if creators else "DataCite"
            year = str(attrs.get("publicationYear", "?"))
            return {
                "provider": "DataCite",
                "doi": doi,
                "year": year,
                "first_author": first_author,
                "title": re.sub(r"\s+", " ", title).strip(),
            }
    except urllib.error.HTTPError as e:
        code = e.code
        e.close()
        if code == 404:
            raise ValueError(f"DOI not registered in Crossref or DataCite: {doi}") from e
        raise


def main() -> int:
    records = extract_catalog_dois()
    unique_dois: List[str] = []
    seen: Set[str] = set()
    for _, _, doi in records:
        if doi not in seen:
            seen.add(doi)
            unique_dois.append(doi)

    print(f"Found {len(records)} DOI citations ({len(unique_dois)} unique DOIs) across catalog.")
    failures = 0
    for doi in unique_dois:
        try:
            meta = resolve_doi_metadata(doi)
            print(
                f"[OK - {meta['provider']:<8}] {doi:<32} -> "
                f"{meta['first_author']} ({meta['year']}): {meta['title']}"
            )
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {doi}: {exc}")

    if failures:
        print(f"\nERROR: {failures} DOI(s) failed verification.")
        return 1
    print(f"\nAll {len(unique_dois)} unique DOIs resolved cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
