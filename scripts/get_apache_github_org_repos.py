#!/usr/bin/env python3
"""Fetch GitHub repos for the Apache org and emit a repo list for Scorecard."""
import argparse
import json
import os
import sys
from typing import List, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path


API_BASE = "https://api.github.com"


def load_tokens_from_file(token_file: Path) -> List[str]:
    tokens: List[str] = []
    if not token_file.exists():
        return tokens
    for raw in token_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                tokens.append(value)
        else:
            tokens.append(line.strip('"').strip("'"))
    return tokens


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "apache-repo-fetcher",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            start = part.find("<")
            end = part.find(">")
            if start != -1 and end != -1 and end > start:
                return part[start + 1:end]
    return None


def fetch_all_repos(org: str, token: Optional[str]) -> List[dict]:
    repos: List[dict] = []
    url = f"{API_BASE}/orgs/{org}/repos?per_page=100&type=public"
    while url:
        req = Request(url, headers=_auth_headers(token))
        try:
            with urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                repos.extend(data)
                url = _parse_next_link(resp.headers.get("Link"))
        except (HTTPError, URLError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")
    return repos


def main() -> None:
    ap = argparse.ArgumentParser(description="List GitHub repos for the Apache org.")
    ap.add_argument("--org", default="apache", help="GitHub org (default: apache).")
    ap.add_argument("--output-dir", default="apache", help="Output directory (default: apache).")
    ap.add_argument("--output", default=None, help="Output file path (default: <output-dir>/apache_github_org_repos.txt).")
    ap.add_argument("--token-file", default=".scorecard_tokens.env", help="Token file (optional).")
    ap.add_argument("--env-token", default="GITHUB_TOKEN", help="Env var token fallback (default: GITHUB_TOKEN).")
    args = ap.parse_args()

    token = None
    token_file = Path(args.token_file).expanduser()
    tokens = load_tokens_from_file(token_file)
    if tokens:
        token = tokens[0]
    else:
        token = os.environ.get(args.env_token)

    repos = fetch_all_repos(args.org, token)
    repo_urls = sorted({r.get("html_url") for r in repos if isinstance(r.get("html_url"), str)})

    output_path = args.output
    if not output_path:
        output_path = f"{args.output_dir}/{args.org}_github_org_repos.txt"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(repo_urls) + ("\n" if repo_urls else ""), encoding="utf-8")

    sys.stderr.write(f"Org: {args.org}\n")
    sys.stderr.write(f"Repos (deduped): {len(repo_urls)}\n")
    sys.stderr.write(f"Output: {output_path}\n")


if __name__ == "__main__":
    main()
