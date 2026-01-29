#!/usr/bin/env python3
"""
Run OpenSSF Scorecard on a list of repos and persist results under apache/.
"""
import argparse
import json
import os
import subprocess
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class TokenRotator:
    """Manages multiple GitHub tokens with rate limiting."""

    def __init__(self, tokens: List[str], requests_per_hour: int = 1900):
        self.tokens = tokens
        self.requests_per_hour = requests_per_hour
        self.current_index = 0
        self.request_times = defaultdict(list)
        self.lock = Lock()

    def get_token(self) -> str:
        with self.lock:
            for _ in range(len(self.tokens)):
                token = self.tokens[self.current_index]
                cutoff = datetime.now() - timedelta(hours=1)
                self.request_times[token] = [
                    t for t in self.request_times[token] if t > cutoff
                ]
                if len(self.request_times[token]) < self.requests_per_hour:
                    self.request_times[token].append(datetime.now())
                    self.current_index = (self.current_index + 1) % len(self.tokens)
                    return token
                self.current_index = (self.current_index + 1) % len(self.tokens)
            time.sleep(60)
            return self.get_token()


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


def read_repo_list(path: Path) -> List[str]:
    repos: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def parse_owner_repo(repo_url: str) -> Optional[str]:
    cleaned = repo_url.strip().rstrip("/")
    if "github.com/" not in cleaned:
        return None
    return cleaned.split("github.com/", 1)[1]


def fetch_repo_metadata(repo_url: str, token: str) -> Optional[Dict]:
    owner_repo = parse_owner_repo(repo_url)
    if not owner_repo:
        return None
    api_url = f"https://api.github.com/repos/{owner_repo}"
    req = Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "scorecard-list-runner",
        },
    )
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError):
        return None


def run_scorecard(repo_url: str, token: str, timeout: int = 300) -> Optional[Dict]:
    try:
        result = subprocess.run(
            ["scorecard", f"--repo={repo_url}", "--format=json"],
            env={**os.environ, "GITHUB_AUTH_TOKEN": token},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return None
    except json.JSONDecodeError:
        return None
    except FileNotFoundError:
        raise RuntimeError(
            "scorecard command not found. Install with `brew install scorecard` or "
            "`go install github.com/ossf/scorecard/v5/cmd/scorecard@latest`."
        )


def process_one_repo(
    repo_url: str,
    token_rotator: TokenRotator,
    out_dir: Path,
    results_file: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    safe_name = repo_url.replace("https://", "").replace("http://", "").replace("/", "_")
    out_path = out_dir / f"{safe_name}.scorecard.json"

    if out_path.exists() and not overwrite:
        return False, "skip_exists"

    token = token_rotator.get_token()
    meta = fetch_repo_metadata(repo_url, token)
    data = run_scorecard(repo_url, token)
    if not data:
        return False, "scorecard_failed"

    payload = {
        "repo": repo_url,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_metadata": None,
        "data": data,
    }
    if meta:
        payload["repo_metadata"] = {
            "name": meta.get("name"),
            "full_name": meta.get("full_name"),
            "html_url": meta.get("html_url"),
            "description": meta.get("description"),
            "default_branch": meta.get("default_branch"),
            "archived": meta.get("archived"),
            "fork": meta.get("fork"),
            "stargazers_count": meta.get("stargazers_count"),
        }
    if not dry_run:
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return True, "success"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run OpenSSF Scorecard on a repo list.")
    ap.add_argument("--repo-list", required=True, help="Path to a text file with repo URLs.")
    ap.add_argument("--out-dir", default="apache/scorecard", help="Output directory.")
    ap.add_argument("--token-file", default=".scorecard_tokens.env", help="Token file.")
    ap.add_argument("--workers", type=int, default=5, help="Parallel workers.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of repos (0 = no limit).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write outputs.")
    ap.add_argument("--results-file", default="apache/scorecard.results.jsonl", help="Append-only results JSONL.")
    ap.add_argument("--progress-file", default="apache/scorecard.progress.jsonl", help="Progress JSONL file.")
    ap.add_argument("--resume", action="store_true", help="Skip repos already marked success/skip.")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Log every N repos.")
    args = ap.parse_args()

    tokens = load_tokens_from_file(Path(args.token_file).expanduser())
    if not tokens:
        raise RuntimeError("No tokens found in token file.")

    repos = read_repo_list(Path(args.repo_list).expanduser())
    if args.limit and args.limit > 0:
        repos = repos[: args.limit]

    out_dir = Path(args.out_dir).expanduser()
    results_file = Path(args.results_file).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    progress_path = Path(args.progress_file).expanduser()
    if args.resume and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            repo = rec.get("repo")
            status = rec.get("status")
            if isinstance(repo, str) and isinstance(status, str):
                if status in {"success", "skip_exists"}:
                    completed.add(repo)
    if args.resume and completed:
        repos = [r for r in repos if r not in completed]

    print(f"Repos: {len(repos)}")
    print(f"Workers: {args.workers}")
    print(f"Out dir: {out_dir}")
    if args.dry_run:
        print("DRY RUN: no files will be written.")

    token_rotator = TokenRotator(tokens, requests_per_hour=1900)
    success = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_repo = {
            executor.submit(
                process_one_repo,
                repo,
                token_rotator,
                out_dir,
                results_file,
                args.overwrite,
                args.dry_run,
            ): repo
            for repo in repos
        }
        for i, future in enumerate(as_completed(future_to_repo), 1):
            repo = future_to_repo[future]
            try:
                did, status = future.result()
                if did:
                    success += 1
                elif status.startswith("skip"):
                    skipped += 1
                else:
                    errors += 1
            except Exception as e:
                status = f"error: {e}"
                errors += 1

            record = {
                "repo": repo,
                "status": status,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            if i % args.checkpoint_every == 0:
                print(
                    f"[checkpoint] processed={i} success={success} skipped={skipped} errors={errors}"
                )

    print("Done.")
    print(f"success={success} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
