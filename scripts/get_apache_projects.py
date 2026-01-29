#!/usr/bin/env python3
"""Fetch Apache Foundation projects and emit a repo list for Scorecard."""
import argparse
import json
import sys
from typing import Dict, List, Iterable, Tuple
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

DEFAULT_URL = "https://projects.apache.org/json/foundation/projects.json"


def _iter_repo_values(repo_field) -> Iterable[str]:
    if repo_field is None:
        return []
    if isinstance(repo_field, str):
        return [repo_field]
    if isinstance(repo_field, list):
        vals: List[str] = []
        for item in repo_field:
            if isinstance(item, str):
                vals.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, str):
                        vals.append(v)
        return vals
    if isinstance(repo_field, dict):
        return [v for v in repo_field.values() if isinstance(v, str)]
    return []


def _is_github(url: str) -> bool:
    return "github.com" in url.lower()


def _normalize_repo(url: str) -> str:
    return url.strip()


def fetch_projects(url: str) -> Dict[str, dict]:
    try:
        with urlopen(url) as resp:
            data = resp.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")
    return json.loads(data)


def collect_repos(projects: Dict[str, dict], include_non_github: bool) -> Tuple[List[str], List[dict]]:
    repos: List[str] = []
    rows: List[dict] = []
    for _, meta in projects.items():
        name = meta.get("name") or meta.get("shortdesc") or "unknown"
        repo_field = meta.get("repository") or meta.get("repo") or meta.get("repos")
        repo_values = [_normalize_repo(r) for r in _iter_repo_values(repo_field)]
        if not include_non_github:
            repo_values = [r for r in repo_values if _is_github(r)]
        repo_values = [r for r in repo_values if r]
        if not repo_values:
            continue
        repos.extend(repo_values)
        rows.append({"project": name, "repos": repo_values})
    repos = sorted(set(repos))
    return repos, rows


def write_output(repos: List[str], rows: List[dict], fmt: str, output: str | None) -> None:
    if fmt == "txt":
        payload = "\n".join(repos) + ("\n" if repos else "")
    elif fmt == "json":
        payload = json.dumps(rows, indent=2)
    elif fmt == "jsonl":
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="List Apache Foundation project repos for Scorecard.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Apache projects JSON URL.")
    ap.add_argument("--output-dir", default="apache", help="Output directory (default: apache).")
    ap.add_argument("--output", default=None, help="Output file path (default: <output-dir>/apache_repos.txt).")
    ap.add_argument("--format", choices=["txt", "json", "jsonl"], default="txt", help="Output format.")
    ap.add_argument(
        "--include-non-github",
        action="store_true",
        help="Include non-GitHub repos (default: only GitHub repos).",
    )
    args = ap.parse_args()

    projects = fetch_projects(args.url)
    repos, rows = collect_repos(projects, args.include_non_github)

    sys.stderr.write(f"Found {len(projects)} ASF projects\n")
    sys.stderr.write(f"Repos (deduped): {len(repos)}\n")
    if args.include_non_github:
        sys.stderr.write("Including non-GitHub repositories\n")
    else:
        sys.stderr.write("Filtering to GitHub repositories only\n")

    output_path = args.output
    if not output_path:
        output_path = f"{args.output_dir}/apache_repos.{args.format}"
    try:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception:
        pass

    write_output(repos, rows, args.format, output_path)


if __name__ == "__main__":
    main()
