#!/usr/bin/env python3
"""
Run OpenSSF Scorecard on research software repositories with:
- Token rotation (5 tokens, 1900 req/hr each)
- Rate limiting enforcement
- Parallel processing
- Checkpointing and resume capability
- Direct metadata.json integration
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)
class TokenRotator:
    """Manages multiple GitHub tokens with rate limiting."""
    def __init__(self, tokens: List[str], requests_per_hour: int = 1900):
        self.tokens = tokens
        self.requests_per_hour = requests_per_hour
        self.current_index = 0
        self.request_times = defaultdict(list)  # token -> list of request timestamps
        self.lock = Lock()
    def get_token(self) -> str:
        """Get next available token, respecting rate limits."""
        with self.lock:
            # Try each token in rotation
            for _ in range(len(self.tokens)):
                token = self.tokens[self.current_index]
                # Clean old request times (older than 1 hour)
                cutoff = datetime.now() - timedelta(hours=1)
                self.request_times[token] = [
                    t for t in self.request_times[token] if t > cutoff
                ]
                # Check if token has capacity
                if len(self.request_times[token]) < self.requests_per_hour:
                    self.request_times[token].append(datetime.now())
                    next_index = (self.current_index + 1) % len(self.tokens)
                    self.current_index = next_index
                    return token
                # Try next token
                self.current_index = (self.current_index + 1) % len(self.tokens)
            # All tokens exhausted - sleep and retry
            sleep_time = 60  # 1 minute
            print(f"All tokens rate-limited. Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
            return self.get_token()  # Recursive retry
def extract_repo_url(metadata: dict) -> Optional[str]:
    """Extract GitHub URL from metadata."""
    url = metadata.get("url")
    if isinstance(url, str) and "github.com" in url:
        return url.strip()
    data = metadata.get("data", {})
    if isinstance(data, dict):
        html = data.get("html_url")
        if html and "github.com" in html:
            return html.strip()
    return None
def run_scorecard(repo_url: str, token: str, timeout: int = 300) -> Optional[Dict]:
    """
    Run OpenSSF Scorecard on a repository.
    Args:
        repo_url: GitHub repository URL
        token: GitHub token to use
        timeout: Command timeout in seconds
    Returns:
        Scorecard results as dict, or None if failed
    """
    try:
        # Run scorecard command
        result = subprocess.run(
            ["scorecard", f"--repo={repo_url}", "--format=json"],
            env={**os.environ, "GITHUB_AUTH_TOKEN": token},
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"Scorecard failed for {repo_url}: {result.stderr}")
            return None
        # Parse JSON output
        scorecard_data = json.loads(result.stdout)
        return scorecard_data
    except subprocess.TimeoutExpired:
        print(f"Scorecard timeout for {repo_url}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {repo_url}: {e}")
        return None
    except FileNotFoundError:
        print("ERROR: 'scorecard' command not found. Please install it:")
        print("  brew install scorecard  (macOS)")
        print("  go install github.com/ossf/scorecard/v5/cmd/scorecard@latest")
        raise
    except Exception as e:
        print(f"Unexpected error for {repo_url}: {e}")
        return None
def process_one_repo(
    metadata_path: Path,
    token_rotator: TokenRotator,
    overwrite: bool = False,
    dry_run: bool = False
) -> tuple[bool, str]:
    """
    Process a single repository: run scorecard and update metadata.json.
    Returns:
        (success, status_message) tuple
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        return False, f"read_error: {e}"
    # Skip if already has scorecard data
    if not overwrite and "openssf_scorecard" in meta:
        return False, "skip_has_scorecard"
    # Extract repo URL
    repo_url = extract_repo_url(meta)
    if not repo_url or "github.com" not in repo_url:
        return False, "skip_no_github_url"
    # Get token
    token = token_rotator.get_token()
    # Run scorecard
    scorecard_data = run_scorecard(repo_url, token)
    if not scorecard_data:
        return False, "scorecard_failed"
    # Add to metadata
    meta["openssf_scorecard"] = {
        "data": scorecard_data,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Write back
    if not dry_run:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write('\n')
    return True, "success"


def load_tokens_from_file(token_file: Path) -> List[str]:
    """Load tokens from a file. Supports KEY=VALUE or one token per line."""
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
def main():
    parser = argparse.ArgumentParser(
        description="Run OpenSSF Scorecard on research software repositories."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to rseng/software repo (default: current directory)"
    )
    parser.add_argument(
        "--provider",
        choices=["github", "gitlab", "both"],
        default="github",
        help="Which provider to process (default: github)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of repos to process (0 = no limit)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5, one per token)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing scorecard data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes to disk"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Print checkpoint every N repos"
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default=".scorecard_tokens.env",
        help="Path to a token file (ignored by git). Supports KEY=VALUE or one token per line."
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Path to a JSONL file that records completed file paths for resume."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already listed in --progress-file."
    )
    args = parser.parse_args()

    # Load GitHub tokens (file first, then env fallback)
    tokens: List[str] = []
    token_file = Path(args.token_file).expanduser()
    tokens.extend(load_tokens_from_file(token_file))
    if not tokens:
        for i in range(1, 6):
            token = os.getenv(f"GITHUB_TOKEN_{i}")
            if token:
                tokens.append(token)
    if not tokens:
        print("ERROR: No GitHub tokens found.")
        print("Please add tokens to your token file or set GITHUB_TOKEN_1 through GITHUB_TOKEN_5 in .env.")
        print("\nExample:")
        print(f'  # token file: {token_file}')
        print('  GITHUB_TOKEN_1="ghp_xxxxxxxxxxxxxxxxxxxxx"')
        print('  GITHUB_TOKEN_2="ghp_yyyyyyyyyyyyyyyyyyyyyyy"')
        print("\nOr one token per line:")
        print('  ghp_xxxxxxxxxxxxxxxxxxxxx')
        print('  ghp_yyyyyyyyyyyyyyyyyyyyyyy')
        print("\nOr in .env:")
        print('GITHUB_TOKEN_1="ghp_xxxxxxxxxxxxxxxxxxxxx"')
        print('GITHUB_TOKEN_2="ghp_yyyyyyyyyyyyyyyyyyyyyyy"')
        return
    print(f"Loaded {len(tokens)} GitHub token(s)")
    # Initialize token rotator
    token_rotator = TokenRotator(tokens, requests_per_hour=1900)
    # Find metadata files
    repo_root = Path(args.repo_root).expanduser().resolve()
    db_root = repo_root / "database"
    providers = ["github", "gitlab"] if args.provider == "both" else [args.provider]
    files = []
    for provider in providers:
        root = db_root / provider
        if root.exists():
            files.extend(sorted(root.rglob("metadata.json")))
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    print(f"\nRepo: {repo_root}")
    print(f"Provider(s): {providers}")
    print(f"Files to process: {len(files)}")
    print(f"Workers: {args.workers}")
    if args.progress_file:
        print(f"Progress file: {args.progress_file}")
    if args.dry_run:
        print("DRY RUN: No files will be modified")
    print()

    completed: set[str] = set()
    if args.resume and args.progress_file:
        progress_path = Path(args.progress_file).expanduser()
        if progress_path.exists():
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                path = rec.get("path")
                status = rec.get("status")
                if isinstance(path, str) and isinstance(status, str):
                    if status in {"success", "skip_has_scorecard", "skip_no_github_url"}:
                        completed.add(path)
    if args.resume and completed:
        files = [f for f in files if str(f) not in completed]
        print(f"Resuming: {len(completed)} completed, {len(files)} remaining")
    # Process repos
    success_count = 0
    skip_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(
                process_one_repo,
                path,
                token_rotator,
                args.overwrite,
                args.dry_run
            ): path
            for path in files
        }
        # Process results with progress bar
        with tqdm(total=len(files), desc="Running Scorecard", unit="repo") as pbar:
            for i, future in enumerate(as_completed(future_to_path), 1):
                path = future_to_path[future]
                try:
                    success, status = future.result()
                    if success:
                        success_count += 1
                    elif "skip" in status:
                        skip_count += 1
                    else:
                        error_count += 1
                        if error_count <= 10:  # Show first 10 errors
                            print(f"\n[error] {path.relative_to(repo_root)}: {status}")
                except Exception as e:
                    error_count += 1
                    print(f"\n[error] {path.relative_to(repo_root)}: {e}")
                    status = f"error: {e}"
                if args.progress_file:
                    progress_path = Path(args.progress_file).expanduser()
                    record = {
                        "path": str(path),
                        "status": status,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                pbar.update(1)
                # Checkpoint
                if i % args.checkpoint_every == 0:
                    print(f"\n[checkpoint] processed={i} success={success_count} skipped={skip_count} errors={error_count}")
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Success:  {success_count}")
    print(f"Skipped:  {skip_count}")
    print(f"Errors:   {error_count}")
    print(f"Total:    {len(files)}")
if __name__ == "__main__":
    main()
