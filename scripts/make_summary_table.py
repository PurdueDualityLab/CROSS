#!/usr/bin/env python3
"""
Build a summary table comparing ASF Scorecard results to RS results.
Outputs CSV and LaTeX.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# Scorecard scoring semantics:
# - score == -1 means missing / not applicable / insufficient evidence (e.g., no workflows, no PRs, no releases).
# - score == 0 means evaluated and failed (practice absent). This is different from -1.
# Missingness must not be conflated with failure.

CHECK_ALLOWLIST: List[str] = [
    "Binary-Artifacts",
    "Branch-Protection",
    "CI-Tests",
    "CII-Best-Practices",
    "Code-Review",
    "Contributors",
    "Dangerous-Workflow",
    "Dependency-Update-Tool",
    "Fuzzing",
    "License",
    "Maintained",
    "Packaging",
    "Pinned-Dependencies",
    "SAST",
    "Security-Policy",
    "Signed-Releases",
    "Token-Permissions",
    "Vulnerabilities",
]


def format_median_iqr(series: pd.Series, percent: bool = False) -> str:
    vals = series.dropna().astype(float).to_numpy()
    if vals.size == 0:
        return ""
    med = float(np.nanmedian(vals))
    q1 = float(np.nanpercentile(vals, 25))
    q3 = float(np.nanpercentile(vals, 75))
    if percent:
        med *= 100.0
        q1 *= 100.0
        q3 *= 100.0
        return f"{med:.0f}% ({q1:.0f}–{q3:.0f})"
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


def format_median_iqr_tex(series: pd.Series, percent: bool = False) -> str:
    vals = series.dropna().astype(float).to_numpy()
    if vals.size == 0:
        return ""
    med = float(np.nanmedian(vals))
    q1 = float(np.nanpercentile(vals, 25))
    q3 = float(np.nanpercentile(vals, 75))
    if percent:
        med *= 100.0
        q1 *= 100.0
        q3 *= 100.0
        return f"{med:.0f}\\% ({q1:.0f}--{q3:.0f})"
    return f"{med:.1f} ({q1:.1f}--{q3:.1f})"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta for two samples."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # O(n*m) is OK for moderate sizes; use vectorized comparisons.
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (x.size * y.size)


def _get_scorecard_data(obj: Dict) -> Optional[Dict]:
    sc = obj.get("openssf_scorecard", {}).get("data")
    if isinstance(sc, dict) and "score" in sc and "checks" in sc:
        return sc
    # Some outputs store scorecard payload at top-level "data"
    sc2 = obj.get("data")
    if isinstance(sc2, dict) and "score" in sc2 and "checks" in sc2:
        return sc2
    return None


def parse_asf_jsonl(path: Path) -> Tuple[pd.DataFrame, int, int]:
    rows: List[Dict[str, float]] = []
    loaded = 0
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            loaded += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            sc = _get_scorecard_data(obj)
            if not sc:
                dropped += 1
                continue
            score = sc.get("score")
            checks = sc.get("checks")
            if score is None or not isinstance(checks, list) or len(checks) == 0:
                dropped += 1
                continue
            missing = sum(1 for c in checks if c.get("score") == -1)
            total = len(checks)
            missingness_rate = missing / total if total > 0 else np.nan
            rows.append(
                {
                    "dataset": "ASF",
                    "actor_unit": "ASF-all",
                    "overall_score": float(score),
                    "missingness_rate": float(missingness_rate),
                }
            )
    df = pd.DataFrame(rows)
    return df, loaded, dropped


def parse_rs_csv(path: Path) -> Tuple[pd.DataFrame, int, int]:
    df = pd.read_csv(path)
    if "overall_score" not in df.columns:
        raise RuntimeError(f"RS CSV missing overall_score column. Columns: {list(df.columns)}")
    loaded = len(df)
    # Drop missing scorecard rows
    missing_mask = df["missing_scorecard"].astype(str).str.lower().isin(["yes", "true", "1"])
    dropped = int(missing_mask.sum())
    df = df.loc[~missing_mask].copy()

    df["actor_unit"] = df["actor_unit"].fillna("Unknown")
    df["actor_unit"] = df["actor_unit"].replace("", "Unknown")

    check_cols = [c for c in CHECK_ALLOWLIST if c in df.columns]
    # Convert check columns to numeric; coerce errors to NaN
    for c in check_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    overall = pd.to_numeric(df["overall_score"], errors="coerce")
    df["overall_score"] = overall

    def row_missingness(row: pd.Series) -> float:
        values = row[check_cols]
        total_present = values.notna().sum()
        if total_present == 0:
            return np.nan
        missing_count = (values == -1).sum()
        return float(missing_count / total_present)

    df["missingness_rate"] = df.apply(row_missingness, axis=1)
    df = df.dropna(subset=["missingness_rate"])
    df = df[["actor_unit", "overall_score", "missingness_rate"]].copy()
    df["dataset"] = "RS"
    return df, loaded, dropped


def summarize_group(df: pd.DataFrame) -> Dict[str, object]:
    if "overall_score" not in df.columns:
        return {
            "n": 0,
            "score_median_iqr": "",
            "missing_median_iqr": "",
            "score_median": np.nan,
        }
    return {
        "n": int(df["overall_score"].notna().sum()),
        "score_median_iqr": format_median_iqr(df["overall_score"]),
        "missing_median_iqr": format_median_iqr(df["missingness_rate"], percent=True),
        "score_median": float(np.nanmedian(df["overall_score"].to_numpy()))
        if df["overall_score"].notna().any()
        else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create ASF vs RS summary table.")
    ap.add_argument("--rs", default="scorecard_repo_results.csv", help="RS CSV path.")
    ap.add_argument("--asf", default="apache/scorecard_results.jsonl", help="ASF JSONL path.")
    ap.add_argument("--outdir", default="results", help="Output directory.")
    ap.add_argument("--no-cliffs-delta", action="store_true", help="Disable Cliff's delta computation.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asf_df, asf_loaded, asf_dropped = parse_asf_jsonl(Path(args.asf))
    rs_df, rs_loaded, rs_dropped = parse_rs_csv(Path(args.rs))

    print(f"ASF loaded: {asf_loaded}, dropped: {asf_dropped}, kept: {len(asf_df)}")
    print(f"RS loaded: {rs_loaded}, dropped: {rs_dropped}, kept: {len(rs_df)}")
    if asf_df.empty:
        raise RuntimeError("ASF dataset is empty after parsing. Check the ASF JSONL path/format.")

    # ASF baseline stats
    asf_stats = summarize_group(asf_df)
    asf_score = asf_df["overall_score"].to_numpy()

    # RS groups: all + per actor_unit
    rows: List[Dict[str, object]] = []
    rs_all = summarize_group(rs_df)
    rs_all_delta = rs_all["score_median"] - asf_stats["score_median"]
    cliffs = ""
    if not args.no_cliffs_delta:
        cd = cliffs_delta(rs_df["overall_score"].to_numpy(), asf_score)
        cliffs = f"{cd:.3f}" if not np.isnan(cd) else ""
    rows.append(
        {
            "group_label": "RS: All projects",
            "asf_n": asf_stats["n"],
            "asf_score_median_iqr": asf_stats["score_median_iqr"],
            "asf_missing_median_iqr": asf_stats["missing_median_iqr"],
            "rs_n": rs_all["n"],
            "rs_score_median_iqr": rs_all["score_median_iqr"],
            "rs_missing_median_iqr": rs_all["missing_median_iqr"],
            "delta_median": f"{rs_all_delta:.2f}",
            "cliffs_delta": cliffs,
        }
    )

    actor_counts = rs_df["actor_unit"].value_counts()
    for actor_unit in actor_counts.index:
        subset = rs_df[rs_df["actor_unit"] == actor_unit]
        stats = summarize_group(subset)
        delta = stats["score_median"] - asf_stats["score_median"]
        cliffs = ""
        if not args.no_cliffs_delta:
            cd = cliffs_delta(subset["overall_score"].to_numpy(), asf_score)
            cliffs = f"{cd:.3f}" if not np.isnan(cd) else ""
        rows.append(
            {
                "group_label": f"RS: {actor_unit}",
                "asf_n": asf_stats["n"],
                "asf_score_median_iqr": asf_stats["score_median_iqr"],
                "asf_missing_median_iqr": asf_stats["missing_median_iqr"],
                "rs_n": stats["n"],
                "rs_score_median_iqr": stats["score_median_iqr"],
                "rs_missing_median_iqr": stats["missing_median_iqr"],
                "delta_median": f"{delta:.2f}",
                "cliffs_delta": cliffs,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = outdir / "summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    # LaTeX table
    latex_lines: List[str] = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{OpenSSF Scorecard overall score comparison: ASF baseline vs research software overall and by actor category.}")
    latex_lines.append("\\begin{tabular}{lrrrrrrr}")
    latex_lines.append("\\hline")
    latex_lines.append(
        "Group & ASF N & ASF score (IQR) & ASF missing (IQR) & RS N & RS score (IQR) & RS missing (IQR) & $\\Delta$ median \\\\"
    )
    latex_lines.append("\\hline")

    for row in rows:
        # Reformat percent for LaTeX
        asf_miss = format_median_iqr_tex(
            asf_df["missingness_rate"], percent=True
        )
        rs_label = row["group_label"]
        rs_subset = rs_df if rs_label == "RS: All projects" else rs_df[rs_df["actor_unit"] == rs_label.replace("RS: ", "")]
        rs_miss = format_median_iqr_tex(rs_subset["missingness_rate"], percent=True)
        rs_score = format_median_iqr_tex(rs_subset["overall_score"], percent=False)
        asf_score = format_median_iqr_tex(asf_df["overall_score"], percent=False)
        latex_lines.append(
            f"{row['group_label']} & {row['asf_n']} & {asf_score} & {asf_miss} & {row['rs_n']} & {rs_score} & {rs_miss} & {row['delta_median']} \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append(
        "\\multicolumn{8}{l}{\\footnotesize Notes: Scorecard check scores of -1 indicate missing or not-applicable (insufficient evidence), while 0 indicates an evaluated check that failed. Missingness is the per-repository fraction of checks scored -1 and is summarized by median and IQR.} \\\\"
    )
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    (outdir / "summary_table.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    # Also print notes to stdout
    print(
        "Notes: Scorecard check scores of -1 indicate missing or not-applicable (insufficient evidence), "
        "while 0 indicates an evaluated check that failed. Missingness is the per-repository fraction of "
        "checks scored -1 and is summarized by median and IQR."
    )


if __name__ == "__main__":
    main()
