# Operationalizing Research Software for Supply Chain Security (Artifact)


This repository contains the code and analyses for **"Operationalizing Research Software for Supply Chain Security"** by Kalu et al.

It is built on top of the **Research Software Engineering (rseng)** and **uses the rseng/software database**. We credit and acknowledge the rseng repositories and database that provide the underlying data ([![DOI](https://zenodo.org/badge/268308501.svg)](https://zenodo.org/badge/latestdoi/268308501), (https://github.com/rseng/software/tree/1.0.0)). 

Our work uses this foundation data to propose a Research Software Supply Chain based Taxonomy for Research Software. Then we tested this taxonomy with OpenSSF scorecard (collection and analysis scripts), We also collect a benchmark database of Apache Software Foundation (ASF) project repositories (to which we also obtained the OpenSSF Scorecard reports), and finally we present summary tables to highlight our results.



## Our Contributions  (scripts and analyses)

All project-specific work for this paper lives in `scripts/`. Below is what each script does and how to run it.

### 1) Batch taxonomy annotation (GPT 5.1)


What batch annotation does:
- Reads `database/github/**/metadata.json` and applies the RSSC prompt (defined inside `scripts/annotate_db_gpt.py`).
- Writes results to `New_SSC_Taxonomy.gpt-5.1` inside each metadata file.
- `--batch-size` + `--batch-index` let you process the dataset in chunks.
- `--auto-batch` runs all chunks sequentially.
- `--progress-file` + `--resume` skip completed entries and allow restart-safe runs.
For large runs of the GPT annotator, `scripts/annotate_db_gpt.py` supports batching and resumable progress logs.

```bash
# process a single batch
python scripts/annotate_db_gpt.py --batch-size 100 --batch-index 0

# process all batches sequentially
python scripts/annotate_db_gpt.py --batch-size 100 --auto-batch

# record progress and resume safely after failures
python scripts/annotate_db_gpt.py --batch-size 100 --auto-batch \
  --progress-file /tmp/annotate.progress.jsonl --resume
```

### 2) OpenSSF Scorecard collection

- `scripts/scorecard_runner.py`  
  Runs OpenSSF Scorecard over the rseng database and writes results into each repo’s `metadata.json` under `openssf_scorecard`.
  ```bash
  python scripts/scorecard_runner.py --token-file .scorecard_tokens.env --progress-file .scorecard.progress.jsonl --resume
  ```

- `scripts/run_scorecard_list.py`  
  Runs Scorecard over a **list of repo URLs** and writes JSON outputs under `apache/scorecard/` plus an append-only JSONL at `apache/scorecard.results.jsonl`.
  ```bash
  python scripts/run_scorecard_list.py --repo-list apache/apache_github_org_repos.txt --token-file .scorecard_tokens.env --resume
  ```

### 3) Apache repository list builders

- `scripts/get_apache_projects.py`  
  Fetches ASF project registry and writes a GitHub repo list to `apache/apache_repos.txt` by default.
  ```bash
  python scripts/get_apache_projects.py
  ```

- `scripts/get_apache_github_org_repos.py`  
  Lists all repositories in the **apache** GitHub org and writes `apache/apache_github_org_repos.txt`.
  ```bash
  python scripts/get_apache_github_org_repos.py
  ```

### 4) Database and analysis outputs

- `scripts/db_results.py`  
  Builds:
  - `scorecard_by_actor.csv` (aggregated stats by actor_unit)
  - `scorecard_missing.csv` (repos missing scorecard and/or taxonomy)
  - `scorecard_repo_results.csv` (per-repo rows with taxonomy + scorecard scores + created_at)
  ```bash
  python scripts/db_results.py --repo-root . --repo-output scorecard_repo_results.csv
  ```

- `scripts/summarize_repo_results.py`  
  Summarizes the per-repo CSV into:
  - `taxonomy_category_percentages.csv`
  - `scorecard_by_distribution_pathway.csv`
  - `taxonomy_sankey.csv` (Sankey links as CSV)
  ```bash
  python scripts/summarize_repo_results.py --input-csv scorecard_repo_results.csv --exclude-na
  ```
  `--exclude-na` drops `-1` scores (not applicable / insufficient evidence) from averages while keeping `0` (evaluated and failed).

- `scripts/make_summary_table.py`  
  Produces a combined ASF vs RS summary table:
  - `results/summary_table.csv`
  - `results/summary_table.tex`
  ```bash
  python scripts/make_summary_table.py \
    --rs scorecard_repo_results.csv \
    --asf apache/scorecard.results.jsonl \
    --outdir results/
  ```

## Outputs (CSV / Sankey)

- `scorecard_repo_results.csv` — per-repo taxonomy + scorecard scores (rseng database)
- `taxonomy_category_percentages.csv` — taxonomy distribution table
- `scorecard_by_distribution_pathway.csv` — average scores by distribution_pathway
- `taxonomy_sankey.csv` — Sankey links (source, target, value)
- `results/summary_table.csv` / `results/summary_table.tex` — ASF vs RS comparisons

## Tokens and configuration

### GitHub tokens (Scorecard)
Create `.scorecard_tokens.env` (ignored by git) with one token per line or KEY=VALUE:
```
GITHUB_TOKEN_1="ghp_..."
GITHUB_TOKEN_2="ghp_..."
```

### OpenAI tokens (GPT annotation)
Set `OPENAI_API_KEY` in `.env`:
```
OPENAI_API_KEY="sk-..."
```

## Installing dependencies

### OpenSSF Scorecard CLI
```
brew install scorecard
# or
go install github.com/ossf/scorecard/v5/cmd/scorecard@latest
```

### Python packages
```
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy tqdm python-dotenv openai
```
