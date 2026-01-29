#!/usr/bin/env python3
"""
Count how many software entries exist in the rseng/software database.

Definition:
- One software tool entry == one metadata.json file anywhere under:
  database/github/**/metadata.json
  database/gitlab/**/metadata.json

This script is robust to varying folder depth and non-standard metadata schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import json
import sys


@dataclass(frozen=True)
class Counts:
    total: int
    by_provider: dict[str, int]


def find_metadata_files(base: Path, provider: str) -> list[Path]:
    root = base / "database" / provider
    if not root.exists():
        return []
    return sorted(root.rglob("metadata.json"))


def guess_owner_from_path(provider_root: Path, metadata_path: Path) -> str | None:
    """
    Try to infer GitHub/GitLab owner/org from the relative path.

    Common layout:
      database/github/<OWNER>/<REPO>/metadata.json
    But sometimes there is extra nesting. We'll take the first path component.
    """
    try:
        rel = metadata_path.relative_to(provider_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    # parts[0] is typically OWNER; parts[-1] is metadata.json
    return parts[0]


def load_uid_if_present(metadata_path: Path) -> str | None:
    """
    Optional: read uid if it exists (can be used to count unique repos if duplicates exist).
    We keep it safe/fast: only parse small JSON and ignore errors.
    """
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        uid = data.get("uid")
        if isinstance(uid, str) and uid.strip():
            return uid.strip()
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Count rseng/software DB entries (metadata.json files).")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the cloned rseng/software repo (default: current directory).",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Print a few sample metadata.json paths per provider.",
    )
    parser.add_argument(
        "--by-owner",
        action="store_true",
        help="Print top owners/orgs by number of entries (provider must be github/gitlab).",
    )
    parser.add_argument(
        "--check-unique-uid",
        action="store_true",
        help="Also count unique 'uid' fields (detect duplicates). Slightly slower.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    db_root = repo_root / "database"
    if not db_root.exists():
        print(f"ERROR: Could not find database/ under {repo_root}", file=sys.stderr)
        return 2

    providers = ["github", "gitlab"]

    all_files: dict[str, list[Path]] = {}
    by_provider_counts: dict[str, int] = {}
    uid_sets: dict[str, set[str]] = defaultdict(set)

    for provider in providers:
        files = find_metadata_files(repo_root, provider)
        all_files[provider] = files
        by_provider_counts[provider] = len(files)

        if args.check_unique_uid:
            for p in files:
                uid = load_uid_if_present(p)
                if uid:
                    uid_sets[provider].add(uid)

    total = sum(by_provider_counts.values())

    print("=== rseng/software database counts ===")
    print(f"Repo root: {repo_root}")
    print(f"Total tools (metadata.json): {total}")
    for provider in providers:
        print(f"  {provider}: {by_provider_counts[provider]}")

    if args.check_unique_uid:
        print("\n=== Unique UID counts (if present) ===")
        for provider in providers:
            u = len(uid_sets.get(provider, set()))
            print(f"  {provider}: {u} unique uid values")
            if u and u != by_provider_counts[provider]:
                print(f"    NOTE: {by_provider_counts[provider]-u} files missing/duplicate uid (inspect if needed)")

    if args.show_samples:
        print("\n=== Sample paths ===")
        for provider in providers:
            files = all_files[provider]
            print(f"\n{provider} samples:")
            for p in files[:10]:
                print(f"  {p.relative_to(repo_root)}")
            if len(files) > 10:
                print(f"  ... ({len(files)-10} more)")

    if args.by_owner:
        print("\n=== Top owners/orgs (by folder prefix) ===")
        for provider in providers:
            provider_root = repo_root / "database" / provider
            files = all_files[provider]
            c = Counter()
            for p in files:
                owner = guess_owner_from_path(provider_root, p)
                if owner:
                    c[owner] += 1
            if not c:
                continue
            print(f"\n{provider}:")
            for owner, cnt in c.most_common(30):
                print(f"  {owner}: {cnt}")

    # Extra: show "deep" nesting count (helps validate your observation)
    deep = 0
    for provider in providers:
        provider_root = repo_root / "database" / provider
        for p in all_files[provider]:
            rel = p.relative_to(provider_root)
            # typical is OWNER/REPO/metadata.json => depth 3
            # anything deeper than 3 indicates extra nesting
            if len(rel.parts) > 3:
                deep += 1
    print(f"\nMetadata paths deeper than OWNER/REPO/metadata.json: {deep}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
