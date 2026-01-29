#!/usr/bin/env python3
"""
Analyze and compare SSC taxonomy classifications from two different models.
Computes agreement statistics, identifies disagreements, and evaluates override effectiveness.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


def load_classified_repos(repo_root: Path, model1: str, model2: str) -> List[Dict[str, Any]]:
    """Load all repos that have been classified by both models."""
    db_root = repo_root / "database" / "github"
    repos = []
    
    for metadata_path in sorted(db_root.rglob("metadata.json")):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Error reading {metadata_path}: {e}")
            continue
        
        if "New_SSC_Taxonomy" not in meta:
            continue
        
        tax = meta["New_SSC_Taxonomy"]
        
        # Check if both models have classified this repo
        if model1 in tax and model2 in tax:
            repos.append({
                "path": str(metadata_path),
                "full_name": meta.get("data", {}).get("full_name", "unknown"),
                "url": meta.get("data", {}).get("html_url", "unknown"),
                "model1": tax[model1],
                "model2": tax[model2],
            })
    
    return repos


def compute_agreement(repos: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """Compute agreement statistics for each classification class."""
    classes = ["actor_unit", "supply_chain_role", "research_role", "distribution_pathway"]
    
    stats = {}
    for cls in classes:
        agree = 0
        disagree = 0
        disagreements = []
        
        for repo in repos:
            val1 = repo["model1"].get(cls)
            val2 = repo["model2"].get(cls)
            
            if val1 == val2:
                agree += 1
            else:
                disagree += 1
                disagreements.append({
                    "repo": repo["full_name"],
                    "url": repo["url"],
                    "model1_value": val1,
                    "model2_value": val2,
                    "model1_overridden": repo["model1"].get("overrides_applied", {}).get(cls, False),
                    "model2_overridden": repo["model2"].get("overrides_applied", {}).get(cls, False),
                })
        
        total = agree + disagree
        agreement_pct = (agree / total * 100) if total > 0 else 0
        
        stats[cls] = {
            "agree": agree,
            "disagree": disagree,
            "total": total,
            "agreement_percentage": agreement_pct,
            "disagreements": disagreements,
        }
    
    return stats


def analyze_overrides(repos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how often overrides were applied and their impact."""
    classes = ["actor_unit", "supply_chain_role", "research_role", "distribution_pathway"]
    
    override_stats = {}
    for cls in classes:
        model1_overrides = 0
        model2_overrides = 0
        both_overridden = 0
        override_caused_agreement = 0
        override_caused_disagreement = 0
        
        for repo in repos:
            m1_override = repo["model1"].get("overrides_applied", {}).get(cls, False)
            m2_override = repo["model2"].get("overrides_applied", {}).get(cls, False)
            
            if m1_override:
                model1_overrides += 1
            if m2_override:
                model2_overrides += 1
            if m1_override and m2_override:
                both_overridden += 1
            
            # Check if override helped or hurt agreement
            if m1_override or m2_override:
                val1 = repo["model1"].get(cls)
                val2 = repo["model2"].get(cls)
                
                if val1 == val2:
                    override_caused_agreement += 1
                else:
                    override_caused_disagreement += 1
        
        override_stats[cls] = {
            "model1_override_count": model1_overrides,
            "model2_override_count": model2_overrides,
            "both_overridden_count": both_overridden,
            "override_led_to_agreement": override_caused_agreement,
            "override_led_to_disagreement": override_caused_disagreement,
        }
    
    return override_stats


def analyze_label_distribution(repos: List[Dict[str, Any]], model_name: str) -> Dict[str, Counter]:
    """Analyze the distribution of labels for each class."""
    classes = ["actor_unit", "supply_chain_role", "research_role", "distribution_pathway"]
    
    distributions = {}
    for cls in classes:
        counter = Counter()
        for repo in repos:
            model_key = "model1" if model_name == "model1" else "model2"
            value = repo[model_key].get(cls)
            if value:
                counter[value] += 1
        distributions[cls] = counter
    
    return distributions


def print_report(repos: List[Dict[str, Any]], model1: str, model2: str):
    """Print comprehensive analysis report."""
    print("="*80)
    print(f"SSC TAXONOMY MODEL COMPARISON REPORT")
    print(f"Model 1: {model1}")
    print(f"Model 2: {model2}")
    print(f"Total Repositories Analyzed: {len(repos)}")
    print("="*80)
    
    # Agreement statistics
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)
    
    agreement_stats = compute_agreement(repos)
    
    for cls, stats in agreement_stats.items():
        print(f"\n{cls.replace('_', ' ').title()}:")
        print(f"  Agreement:    {stats['agree']:3d}/{stats['total']:3d} ({stats['agreement_percentage']:.1f}%)")
        print(f"  Disagreement: {stats['disagree']:3d}/{stats['total']:3d} ({100-stats['agreement_percentage']:.1f}%)")
    
    # Overall agreement
    total_agree = sum(s['agree'] for s in agreement_stats.values())
    total_total = sum(s['total'] for s in agreement_stats.values())
    overall_pct = (total_agree / total_total * 100) if total_total > 0 else 0
    print(f"\n{'Overall Agreement:':<20} {total_agree:3d}/{total_total:3d} ({overall_pct:.1f}%)")
    
    # Override analysis
    print("\n" + "="*80)
    print("OVERRIDE EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    override_stats = analyze_overrides(repos)
    
    for cls, stats in override_stats.items():
        print(f"\n{cls.replace('_', ' ').title()}:")
        print(f"  {model1} overrides:        {stats['model1_override_count']}")
        print(f"  {model2} overrides:        {stats['model2_override_count']}")
        print(f"  Both overridden:           {stats['both_overridden_count']}")
        print(f"  Override → Agreement:      {stats['override_led_to_agreement']}")
        print(f"  Override → Disagreement:   {stats['override_led_to_disagreement']}")
    
    # Label distributions
    print("\n" + "="*80)
    print(f"LABEL DISTRIBUTION - {model1}")
    print("="*80)
    
    dist1 = analyze_label_distribution(repos, "model1")
    for cls, counter in dist1.items():
        print(f"\n{cls.replace('_', ' ').title()}:")
        for label, count in counter.most_common():
            print(f"  {label:<50} {count:3d} ({count/len(repos)*100:5.1f}%)")
    
    print("\n" + "="*80)
    print(f"LABEL DISTRIBUTION - {model2}")
    print("="*80)
    
    dist2 = analyze_label_distribution(repos, "model2")
    for cls, counter in dist2.items():
        print(f"\n{cls.replace('_', ' ').title()}:")
        for label, count in counter.most_common():
            print(f"  {label:<50} {count:3d} ({count/len(repos)*100:5.1f}%)")
    
    # Detailed disagreements
    print("\n" + "="*80)
    print("DETAILED DISAGREEMENTS")
    print("="*80)
    
    for cls, stats in agreement_stats.items():
        if stats['disagreements']:
            print(f"\n{cls.replace('_', ' ').title()} - {len(stats['disagreements'])} disagreements:")
            print("-" * 80)
            
            for i, disagree in enumerate(stats['disagreements'][:10], 1):  # Show first 10
                print(f"\n{i}. {disagree['repo']}")
                print(f"   URL: {disagree['url']}")
                print(f"   {model1}: {disagree['model1_value']}")
                if disagree['model1_overridden']:
                    print(f"            (OVERRIDDEN)")
                print(f"   {model2}: {disagree['model2_value']}")
                if disagree['model2_overridden']:
                    print(f"            (OVERRIDDEN)")
            
            if len(stats['disagreements']) > 10:
                print(f"\n   ... and {len(stats['disagreements']) - 10} more disagreements")


def save_detailed_report(repos: List[Dict[str, Any]], model1: str, model2: str, output_path: Path):
    """Save detailed comparison report as JSON."""
    agreement_stats = compute_agreement(repos)
    override_stats = analyze_overrides(repos)
    dist1 = analyze_label_distribution(repos, "model1")
    dist2 = analyze_label_distribution(repos, "model2")
    
    report = {
        "models_compared": {
            "model1": model1,
            "model2": model2,
        },
        "total_repos": len(repos),
        "agreement_statistics": agreement_stats,
        "override_statistics": override_stats,
        "label_distributions": {
            model1: {cls: dict(counter) for cls, counter in dist1.items()},
            model2: {cls: dict(counter) for cls, counter in dist2.items()},
        },
        "repositories": [
            {
                "full_name": repo["full_name"],
                "url": repo["url"],
                "classifications": {
                    model1: repo["model1"],
                    model2: repo["model2"],
                }
            }
            for repo in repos
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SSC taxonomy classifications between two models."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to rseng/software repo (default: current directory)"
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="gpt-5.2",
        help="First model name (default: gpt-5.2)"
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="gpt-4o-mini",
        help="Second model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save detailed report to JSON file (optional)"
    )
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root).expanduser().resolve()
    
    print(f"Loading repositories classified by both {args.model1} and {args.model2}...")
    repos = load_classified_repos(repo_root, args.model1, args.model2)
    
    if not repos:
        print(f"\nNo repositories found with classifications from both models.")
        print(f"Make sure both models have classified some repos.")
        return
    
    print(f"Found {len(repos)} repositories classified by both models.\n")
    
    # Print comprehensive report
    print_report(repos, args.model1, args.model2)
    
    # Save detailed report if requested
    if args.output:
        output_path = Path(args.output)
        save_detailed_report(repos, args.model1, args.model2, output_path)
    
    # Summary recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    agreement_stats = compute_agreement(repos)
    overall_agreement = sum(s['agree'] for s in agreement_stats.values()) / sum(s['total'] for s in agreement_stats.values()) * 100
    
    print(f"\nOverall Agreement: {overall_agreement:.1f}%")
    
    if overall_agreement >= 80:
        print("✅ GOOD: High agreement between models. Heuristics appear effective.")
        print("   Recommendation: Proceed with full dataset classification.")
    elif overall_agreement >= 70:
        print("⚠️  MODERATE: Decent agreement but some inconsistencies.")
        print("   Recommendation: Review disagreements and consider refining heuristics.")
    else:
        print("❌ POOR: Low agreement between models.")
        print("   Recommendation: Review and strengthen heuristics before full run.")
    
    # Check specific problematic classes
    print("\nPer-class assessment:")
    for cls, stats in agreement_stats.items():
        pct = stats['agreement_percentage']
        status = "✅" if pct >= 80 else "⚠️ " if pct >= 70 else "❌"
        print(f"  {status} {cls.replace('_', ' ').title():<25} {pct:5.1f}%")


if __name__ == "__main__":
    main()