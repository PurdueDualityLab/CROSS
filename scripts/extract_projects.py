#!/usr/bin/env python3
"""
Extract N research software projects from the database for OpenSSF Scorecard testing.
Outputs a JSON file with GitHub URLs and basic metadata.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def extract_repo_url(metadata: dict) -> str | None:
    """Extract GitHub URL from metadata."""
    url = metadata.get("url")
    if isinstance(url, str) and url.strip() and "api.github.com" not in url:
        return url.strip()

    data = metadata.get("data", {})
    if isinstance(data, dict):
        html = data.get("html_url")
        if isinstance(html, str) and html.strip():
            return html.strip()

        clone = data.get("clone_url")
        if isinstance(clone, str) and "github.com" in clone and clone.strip():
            clone = clone.strip()
            if clone.endswith(".git"):
                clone = clone[:-4]
            if "api.github.com" not in clone:
                return clone

    return None


def extract_project_info(metadata_path: Path) -> Dict[str, Any] | None:
    """Extract relevant project information from metadata.json."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Error reading {metadata_path}: {e}")
        return None

    repo_url = extract_repo_url(meta)
    if not repo_url or "github.com" not in repo_url:
        return None

    data = meta.get("data", {})
    owner = data.get("owner", {}) if isinstance(data, dict) else {}

    project_info = {
        "github_url": repo_url,
        "uid": meta.get("uid"),
        "full_name": data.get("full_name") if isinstance(data, dict) else None,
        "description": data.get("description") if isinstance(data, dict) else None,
        "language": data.get("language") if isinstance(data, dict) else None,
        "stars": data.get("stargazers_count") if isinstance(data, dict) else None,
        "owner_type": owner.get("type") if isinstance(owner, dict) else None,
        "owner_login": owner.get("login") if isinstance(owner, dict) else None,
        "created_at": data.get("created_at") if isinstance(data, dict) else None,
        "updated_at": data.get("updated_at") if isinstance(data, dict) else None,
    }

    # Include SSC taxonomy if it exists
    if "New_SSC_Taxonomy" in meta:
        project_info["ssc_taxonomy"] = meta["New_SSC_Taxonomy"]

    return project_info


def main():
    parser = argparse.ArgumentParser(
        description="Extract N projects from the database for OpenSSF Scorecard testing."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of projects to extract (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="projects_for_scorecard.json",
        help="Output JSON file (default: projects_for_scorecard.json)"
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
        help="Which provider to extract from (default: github)"
    )
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Output only URLs as text file (one per line)"
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    db_root = repo_root / "database"

    # Collect metadata files
    providers = ["github", "gitlab"] if args.provider == "both" else [args.provider]
    files: List[Path] = []
    
    for provider in providers:
        root = db_root / provider
        if root.exists():
            files.extend(sorted(root.rglob("metadata.json")))

    print(f"Found {len(files)} metadata files")
    print(f"Extracting {args.count} projects...")

    projects = []
    for metadata_path in files:
        if len(projects) >= args.count:
            break

        project = extract_project_info(metadata_path)
        if project:
            projects.append(project)

    print(f"Successfully extracted {len(projects)} projects")

    # Write output
    output_path = Path(args.output)
    
    if args.urls_only:
        # Write URLs only as text file
        if not str(output_path).endswith('.txt'):
            output_path = output_path.with_suffix('.txt')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for project in projects:
                f.write(f"{project['github_url']}\n")
        
        print(f"\nWrote {len(projects)} URLs to {output_path}")
    else:
        # Write full JSON
        if not str(output_path).endswith('.json'):
            output_path = output_path.with_suffix('.json')
        
        output_data = {
            "total_projects": len(projects),
            "extraction_date": "2026-01-21",
            "projects": projects
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nWrote {len(projects)} projects to {output_path}")
        
        # Print sample
        print("\nSample projects:")
        for i, project in enumerate(projects[:3], 1):
            print(f"{i}. {project['full_name']}")
            print(f"   URL: {project['github_url']}")
            print(f"   Language: {project['language']}")
            print(f"   Stars: {project['stars']}")
            if 'ssc_taxonomy' in project:
                print(f"   Has SSC Taxonomy: Yes")
            print()


if __name__ == "__main__":
    main()