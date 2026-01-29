#!/usr/bin/env python3
"""
Summarize per-repo scorecard CSV:
1) Category percentages for taxonomy fields.
2) Scorecard averages by distribution_pathway.
3) Sankey diagram: actor_unit -> supply_chain_role -> research_role -> distribution_pathway.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


TAXONOMY_FIELDS = [
    "actor_unit",
    "supply_chain_role",
    "research_role",
    "distribution_pathway",
]

BASE_FIELDS = {
    "path",
    "repo_url",
    "created_at",
    "missing_scorecard",
    "missing_fields",
    *TAXONOMY_FIELDS,
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_category_percentages(rows: List[Dict[str, str]], output: Path) -> None:
    counts = {field: defaultdict(int) for field in TAXONOMY_FIELDS}
    total = len(rows)
    for row in rows:
        for field in TAXONOMY_FIELDS:
            val = (row.get(field) or "Unknown").strip()
            if not val:
                val = "Unknown"
            counts[field][val] += 1

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["taxonomy_field", "category", "count", "percent"])
        writer.writeheader()
        for field in TAXONOMY_FIELDS:
            for category, count in sorted(counts[field].items(), key=lambda x: (-x[1], x[0])):
                percent = (count / total * 100) if total else 0.0
                writer.writerow(
                    {
                        "taxonomy_field": field,
                        "category": category,
                        "count": count,
                        "percent": round(percent, 2),
                    }
                )


def parse_score_columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cols = [c for c in rows[0].keys() if c not in BASE_FIELDS]
    return sorted(cols)


def safe_float(value: str, exclude_na: bool) -> Tuple[bool, float]:
    if value is None:
        return False, 0.0
    text = str(value).strip()
    if text == "":
        return False, 0.0
    try:
        val = float(text)
        if exclude_na and val == -1:
            return False, 0.0
        return True, val
    except ValueError:
        return False, 0.0


def write_scorecard_by_distribution(rows: List[Dict[str, str]], output: Path, exclude_na: bool) -> None:
    score_cols = parse_score_columns(rows)
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        if row.get("missing_scorecard") == "yes":
            continue
        dist = (row.get("distribution_pathway") or "Unknown").strip() or "Unknown"
        counts[dist] += 1
        for col in score_cols:
            ok, val = safe_float(row.get(col, ""), exclude_na)
            if ok:
                grouped[dist][col].append(val)

    fieldnames = ["distribution_pathway", "repos_with_scorecard"]
    fieldnames.extend([f"{c}_mean" for c in score_cols])

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dist in sorted(counts.keys()):
            row_out: Dict[str, object] = {
                "distribution_pathway": dist,
                "repos_with_scorecard": counts[dist],
            }
            for col in score_cols:
                values = grouped[dist].get(col, [])
                row_out[f"{col}_mean"] = round(sum(values) / len(values), 2) if values else ""
            writer.writerow(row_out)


def build_sankey(rows: List[Dict[str, str]]) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    node_index: Dict[str, int] = {}
    nodes: List[str] = []
    links_count: Dict[Tuple[str, str], int] = defaultdict(int)

    def get_node_id(label: str) -> int:
        if label not in node_index:
            node_index[label] = len(nodes)
            nodes.append(label)
        return node_index[label]

    for row in rows:
        values = []
        for field in TAXONOMY_FIELDS:
            val = (row.get(field) or "Unknown").strip() or "Unknown"
            values.append(f"{field}: {val}")
        for a, b in zip(values, values[1:]):
            links_count[(a, b)] += 1
            get_node_id(a)
            get_node_id(b)

    links: List[Tuple[int, int, int]] = []
    for (a, b), count in links_count.items():
        links.append((node_index[a], node_index[b], count))
    return nodes, links


def write_sankey_csv(nodes: List[str], links: List[Tuple[int, int, int]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "value"])
        writer.writeheader()
        for s, t, v in links:
            writer.writerow({"source": nodes[s], "target": nodes[t], "value": v})


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize per-repo scorecard CSV.")
    ap.add_argument("--input-csv", default="scorecard_repo_results.csv", help="Input CSV from db_results.py.")
    ap.add_argument("--category-output", default="taxonomy_category_percentages.csv", help="Category percentages CSV.")
    ap.add_argument("--distribution-output", default="scorecard_by_distribution_pathway.csv", help="Scorecard summary CSV.")
    ap.add_argument("--sankey-output", default="taxonomy_sankey.csv", help="Sankey links CSV output.")
    ap.add_argument(
        "--exclude-na",
        action="store_true",
        help="Exclude -1 (not applicable / cannot be evaluated) from score averages.",
    )
    args = ap.parse_args()

    rows = read_rows(Path(args.input_csv))
    if not rows:
        raise RuntimeError("No rows found in input CSV.")

    write_category_percentages(rows, Path(args.category_output))
    write_scorecard_by_distribution(rows, Path(args.distribution_output), args.exclude_na)

    nodes, links = build_sankey(rows)
    write_sankey_csv(nodes, links, Path(args.sankey_output))

    print(f"Wrote: {args.category_output}")
    print(f"Wrote: {args.distribution_output}")
    print(f"Wrote: {args.sankey_output}")


if __name__ == "__main__":
    main()
