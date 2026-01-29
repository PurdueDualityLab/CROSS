#!/usr/bin/env python3
"""
Analyze OpenSSF Scorecard results grouped by actor unit.
Reads metadata.json files and creates CSV with aggregated statistics.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics
import csv
def extract_actor_unit(metadata: dict) -> str:
    """Extract actor unit from New_SSC_Taxonomy."""
    try:
        return metadata.get("New_SSC_Taxonomy", {}).get("gpt-5.1", {}).get("actor_unit", "Unknown")
    except (AttributeError, KeyError):
        return "Unknown"
def extract_scorecard_scores(metadata: dict) -> Dict[str, float]:
    """
    Extract all check scores from OpenSSF Scorecard data.
    Returns dict of {check_name: score} with -1 handled as None.
    """
    scores = {}
    try:
        checks = metadata.get("openssf_scorecard", {}).get("data", {}).get("checks", [])
        overall_score = metadata.get("openssf_scorecard", {}).get("data", {}).get("score")
        # Add overall score
        if overall_score is not None:
            scores["overall_score"] = overall_score
        # Add individual check scores
        for check in checks:
            name = check.get("name", "")
            score = check.get("score", -1)
            # Store score (-1 will be filtered later)
            if name:
                scores[name] = score
    except (AttributeError, KeyError, TypeError):
        pass
    return scores


def extract_taxonomy_fields(metadata: dict) -> Dict[str, str]:
    tax = metadata.get("New_SSC_Taxonomy", {}).get("gpt-5.1", {}) or {}
    return {
        "actor_unit": tax.get("actor_unit", "Unknown"),
        "supply_chain_role": tax.get("supply_chain_role", "Unknown"),
        "research_role": tax.get("research_role", "Unknown"),
        "distribution_pathway": tax.get("distribution_pathway", "Unknown"),
    }


def missing_taxonomy_fields(taxonomy: Dict[str, str]) -> List[str]:
    missing = []
    for key, value in taxonomy.items():
        if value in (None, "", "Unknown"):
            missing.append(key)
    return missing
def main():
    parser = argparse.ArgumentParser(
        description="Analyze OpenSSF Scorecard results by actor unit."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to software repo (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scorecard_by_actor.csv",
        help="Output CSV filename (default: scorecard_by_actor.csv)"
    )
    parser.add_argument(
        "--exclude-na",
        action="store_true",
        help="Exclude -1 (N/A) scores from calculations"
    )
    parser.add_argument(
        "--missing-output",
        type=str,
        default="scorecard_missing.csv",
        help="CSV listing repos missing scorecard and/or taxonomy fields"
    )
    parser.add_argument(
        "--repo-output",
        type=str,
        default="scorecard_repo_results.csv",
        help="Per-repo CSV with taxonomy and scorecard scores"
    )
    args = parser.parse_args()
    # Find all metadata.json files
    repo_root = Path(args.repo_root).expanduser().resolve()
    db_root = repo_root / "database" / "github"
    if not db_root.exists():
        print(f"ERROR: {db_root} does not exist")
        return
    metadata_files = list(db_root.rglob("metadata.json"))
    print(f"Found {len(metadata_files)} metadata.json files")
    # Data structure: {actor_unit: {check_name: [scores]}}
    data = defaultdict(lambda: defaultdict(list))
    missing_rows: List[Dict[str, str]] = []
    repo_rows: List[Dict[str, str]] = []
    all_repo_checks: set[str] = set()
    # Process each file
    repos_with_scorecard = 0
    repos_with_taxonomy = 0
    for filepath in metadata_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            taxonomy = extract_taxonomy_fields(metadata)
            actor_unit = taxonomy["actor_unit"]
            missing_fields = missing_taxonomy_fields(taxonomy)
            if actor_unit != "Unknown":
                repos_with_taxonomy += 1
            # Check if has scorecard data
            missing_scorecard = "openssf_scorecard" not in metadata
            created_at = (
                metadata.get("data", {}).get("created_at")
                or metadata.get("created_at")
                or ""
            )
            base_row = {
                "path": str(filepath.relative_to(repo_root)),
                "repo_url": metadata.get("url", ""),
                "created_at": created_at,
                "missing_scorecard": "yes" if missing_scorecard else "no",
                "missing_fields": ",".join(missing_fields) if missing_fields else "",
                **taxonomy,
            }
            if missing_fields or missing_scorecard:
                missing_rows.append(base_row)
            if missing_scorecard:
                repo_rows.append(base_row)
                continue
            repos_with_scorecard += 1
            # Extract scores
            scores = extract_scorecard_scores(metadata)
            all_repo_checks.update(scores.keys())
            # Add to data structure
            for check_name, score in scores.items():
                # Optionally exclude -1 (N/A) scores
                if args.exclude_na and score == -1:
                    continue
                data[actor_unit][check_name].append(score)
            row_with_scores = {**base_row, **scores}
            repo_rows.append(row_with_scores)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    print(f"Repos with scorecard data: {repos_with_scorecard}")
    print(f"Repos with taxonomy classification: {repos_with_taxonomy}")
    # Get all unique check names
    all_checks = set()
    for actor_data in data.values():
        all_checks.update(actor_data.keys())
    all_checks = sorted(all_checks)
    # Calculate statistics and write CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        # Create column headers
        fieldnames = ['actor_unit', 'total_repos', 'missing_scorecard_count']
        for check in all_checks:
            fieldnames.extend([
                f"{check}_mean",
                f"{check}_median",
                f"{check}_count"
            ])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Write data for each actor unit
        for actor_unit in sorted(data.keys()):
            missing_count = sum(
                1
                for m in missing_rows
                if m["actor_unit"] == actor_unit and m["missing_scorecard"] == "yes"
            )
            row = {
                'actor_unit': actor_unit,
                'total_repos': len(set(
                    i for check_scores in data[actor_unit].values()
                    for i in range(len(check_scores))
                )),
                'missing_scorecard_count': missing_count,
            }
            for check in all_checks:
                scores = data[actor_unit].get(check, [])
                if scores:
                    # Filter out -1 if needed
                    if args.exclude_na:
                        valid_scores = [s for s in scores if s != -1]
                    else:
                        valid_scores = scores
                    if valid_scores:
                        row[f"{check}_mean"] = round(statistics.mean(valid_scores), 2)
                        row[f"{check}_median"] = round(statistics.median(valid_scores), 2)
                        row[f"{check}_count"] = len(valid_scores)
                    else:
                        row[f"{check}_mean"] = ""
                        row[f"{check}_median"] = ""
                        row[f"{check}_count"] = 0
                else:
                    row[f"{check}_mean"] = ""
                    row[f"{check}_median"] = ""
                    row[f"{check}_count"] = 0
            writer.writerow(row)
    print(f"\nCreated {args.output}")
    print(f"Actor units found: {len(data)}")
    print(f"Checks analyzed: {len(all_checks)}")
    # Write missing scorecard/taxonomy CSV
    with open(args.missing_output, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "path",
            "repo_url",
            "created_at",
            "missing_scorecard",
            "actor_unit",
            "supply_chain_role",
            "research_role",
            "distribution_pathway",
            "missing_fields",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in missing_rows:
            writer.writerow(row)
    print(f"Missing scorecard/taxonomy list: {args.missing_output} ({len(missing_rows)})")

    # Write per-repo results CSV
    repo_fieldnames = [
        "path",
        "repo_url",
        "created_at",
        "missing_scorecard",
        "actor_unit",
        "supply_chain_role",
        "research_role",
        "distribution_pathway",
        "missing_fields",
    ]
    repo_fieldnames.extend(sorted(all_repo_checks))
    with open(args.repo_output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=repo_fieldnames)
        writer.writeheader()
        for row in repo_rows:
            writer.writerow(row)
    print(f"Per-repo results: {args.repo_output} ({len(repo_rows)})")
if __name__ == "__main__":
    main()
