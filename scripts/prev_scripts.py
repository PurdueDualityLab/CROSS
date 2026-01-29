"""
these are the previous iterations of the scripts that are outdated but just kept for records
"""

# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import time
# from pathlib import Path
# from typing import List, Optional, Literal

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# # Always load .env from repo root and override any existing shell env var
# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Build and release software",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]

# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Build and release software",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional extra detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="0 to 1 confidence score in the labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Try common places for a GitHub URL.
#     """
#     url = metadata.get("url")
#     if isinstance(url, str) and url.strip():
#         return url.strip()

#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         for k in ("html_url", "clone_url", "url"):
#             v = data.get(k)
#             if isinstance(v, str) and "github.com" in v:
#                 return v.strip()

#     return None


# # ----------------------------
# # README fetching
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations. Using raw URLs avoids GitHub API rate limits.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Prompting / GPT call
# # ----------------------------
# def build_prompt(readme: str) -> str:
#     return f"""
# You are labeling a software repository using ONLY the repository README as evidence.

# Task:
# Return exactly one label for each dimension:
# 1) actor_unit (who can change it / enforce controls)
# 2) supply_chain_role (where it sits in the research software supply chain)
# 3) research_role (how directly it contributes to producing research results)
# 4) distribution_pathway (how it is delivered to downstream users)

# Allowed actor_unit values:
# {json.dumps(ACTOR_UNIT_VALUES, indent=2)}

# Allowed supply_chain_role values:
# {json.dumps(SUPPLY_CHAIN_ROLE_VALUES, indent=2)}

# Allowed research_role values:
# {json.dumps(RESEARCH_ROLE_VALUES, indent=2)}

# Allowed distribution_pathway values:
# {json.dumps(DISTRIBUTION_PATHWAY_VALUES, indent=2)}

# Decision rules (follow strictly):

# ACTOR UNIT:
# - Individual maintainer: personal-maintained, no institutional/community governance signs.
# - Research group or lab: tied to a specific lab/project team/PI/students; lab identity present.
# - Institution: university/national lab/institute/government org appears responsible/hosting/operating.
# - Community or foundation: broader open-source governance beyond one lab/institution; formal processes/roles.
# - Vendor or commercial entity: company primarily develops/operates it; commercial product/service signals.
# - Platform operator: primary is operating distribution/hosting infra for many projects (registry/CI/hosting).
# - Mixed or shared responsibility: explicit shared responsibility across actor units.
# - Unknown: README does not contain enough evidence.

# SUPPLY CHAIN ROLE:
# - Application software: end-user program to run for research tasks.
# - Dependency software artifact: library/package meant to be imported/depended upon.
# - Build and release software: CI/build/test/package/sign/publish tooling.
# - Infrastructure: provides environment/substrate (orchestration/runtime/platform) where software runs.
# - Runtime instrumentation software: profiling/monitoring/instrumentation/modification at runtime.
# - Assistant layer software: interactive assistance in workflows (chatbots / guided automation).
# - Unknown: insufficient README evidence.

# RESEARCH ROLE (Research coupling):
# - Direct research execution: directly generates/transforms/analyzes scientific data/models; removing it prevents producing core scientific outputs (results/models/figures).
# - Research-support tooling: enables development, reproducibility, deployment, management of workflows; not the core analysis/model itself.
# - Incidental or general-purpose: generic tool broadly used across domains, not research-specific in intent/design and not clearly part of a research toolchain.
# - Unknown: insufficient evidence.

# DISTRIBUTION PATHWAY:
# - Source repo: README emphasizes clone/build/install from repo; no clear formal release/registry as primary channel.
# - Releases: README emphasizes tagged releases / GitHub Releases / tarballs as consumption unit.
# - Package registry: README shows ecosystem install command (pip/conda/cran/maven/npm/etc). If chosen, set distribution_details to the registry name.
# - Containers: README emphasizes docker/OCI images (docker pull/run). If chosen, set distribution_details (e.g., Docker Hub/GHCR).
# - Installer or binary: README emphasizes downloadable binaries/installers/executables.
# - Network service: README indicates hosted web app/API/managed service where users interact over network rather than installing.
# - Unknown: insufficient evidence from README.

# Output requirements:
# - Output must match the schema exactly.
# - Provide up to 3 short evidence bullets (quotes or paraphrases) drawn from README.
# - Set confidence between 0 and 1.
# - If uncertain, choose "Unknown" and set low confidence.
# - For distribution_pathway="Package registry" or "Containers", set distribution_details (e.g., "PyPI", "conda-forge", "CRAN", "Maven", "npm", "Docker Hub", "GHCR"). Otherwise distribution_details should be null.

# README (verbatim):
# ----------------
# {readme[:120000]}
# ----------------
# """.strip()


# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow labeling rules exactly. Use only README evidence.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     prompt = build_prompt(readme)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": result.actor_unit,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": result.research_role,
#         "distribution_pathway": result.distribution_pathway,
#         "distribution_details": result.distribution_details,
#         "evidence": result.evidence[:3],
#         "confidence": float(result.confidence),
#         "model": model,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     metadata_path.write_text(
#         json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#         encoding="utf-8",
#     )

#     time.sleep(delay)
#     return True, "updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo (default: current directory).")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name (try gpt-5.1 and a mini model).")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing New_SSC_Taxonomy field.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Print a checkpoint log every N processed files.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (repo root) or export it in your shell.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     providers: List[str]
#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()

##!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import List, Optional, Literal, Tuple

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# # Always load .env from repo root and override any existing shell env var
# # (Run from repo root so ".env" is the right file.)
# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Build and release software",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]


# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Build and release software",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional extra detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="0 to 1 confidence score in the labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Prefer canonical GitHub HTML URL, and avoid api.github.com URLs.
#     """
#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         html_url = data.get("html_url")
#         if isinstance(html_url, str) and html_url.startswith("https://github.com/"):
#             return html_url.rstrip("/")

#     url = metadata.get("url")
#     if isinstance(url, str) and url.startswith("https://github.com/") and "api.github.com" not in url:
#         return url.rstrip("/")

#     if isinstance(data, dict):
#         clone = data.get("clone_url")
#         if isinstance(clone, str) and clone.startswith("https://github.com/"):
#             # https://github.com/org/repo.git -> https://github.com/org/repo
#             return clone.replace(".git", "").rstrip("/")

#     return None


# def repo_context_for_prompt(meta: dict) -> dict:
#     """
#     Small, stable repo metadata block to help actor_unit classification.
#     This is NOT meant to replace README; it's only extra grounding.
#     """
#     data = meta.get("data", {}) if isinstance(meta.get("data"), dict) else {}
#     owner = data.get("owner", {}) if isinstance(data.get("owner"), dict) else {}

#     ctx = {
#         "uid": meta.get("uid"),
#         "full_name": data.get("full_name") or data.get("name"),
#         "html_url": data.get("html_url") or meta.get("url"),
#         "owner_login": owner.get("login"),
#         "owner_type": owner.get("type"),  # "User" or "Organization"
#         "description": data.get("description"),
#         "homepage": data.get("homepage"),
#         "topics": data.get("topics", []),
#     }
#     return ctx


# # ----------------------------
# # README fetching
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations. Using raw URLs avoids GitHub API rate limits.
#     NOTE: still imperfect; we are intentionally NOT adding GitHub-token API fallback per your request.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Distribution pathway detection (deterministic override)
# # ----------------------------
# def detect_distribution(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
#     """
#     Returns (distribution_pathway, distribution_details, evidence_snippets)

#     We keep this simple + conservative. If we detect a clear registry/container signal,
#     we override the model's distribution_* fields to reduce flip-flopping.
#     """
#     t = readme.lower()
#     ev: List[str] = []

#     def add_ev(s: str) -> None:
#         if s and s not in ev and len(ev) < 3:
#             ev.append(s)

#     # Containers
#     if re.search(r"\bdocker\s+pull\b|\bdocker\s+run\b|ghcr\.io|dockerhub", t):
#         details = "GHCR" if "ghcr.io" in t else "Docker Hub"
#         add_ev("Detected container usage (docker pull/run or GHCR/Docker Hub reference).")
#         return "Containers", details, ev

#     # Package registries
#     if re.search(r"\bpip3?\s+install\b", t) or "pip install" in t:
#         add_ev("Detected pip install command.")
#         return "Package registry", "PyPI", ev

#     if re.search(r"\bconda\s+install\b", t):
#         details = "conda-forge" if "conda-forge" in t else "conda"
#         add_ev("Detected conda install command.")
#         return "Package registry", details, ev

#     if "pkg.add(" in t or "pkg.add(" in t.replace(" ", ""):
#         add_ev("Detected Julia Pkg.add(...) install.")
#         return "Package registry", "Julia General registry", ev

#     if re.search(r"\bnpm\s+install\b", t):
#         add_ev("Detected npm install command.")
#         return "Package registry", "npm", ev

#     if re.search(r"\bmvn\b|\bgradle\b|\bimplementation\s+['\"]", t):
#         add_ev("Detected Maven/Gradle dependency installation cues.")
#         return "Package registry", "Maven/Gradle", ev

#     # Releases vs binaries (very rough)
#     if "github releases" in t or ("release" in t and "download" in t):
#         add_ev("Detected GitHub Releases / download wording.")
#         return "Releases", "GitHub Releases", ev

#     return None, None, []


# # ----------------------------
# # Prompting / GPT call
# # ----------------------------
# def build_prompt(readme: str, meta: dict) -> str:
#     ctx = repo_context_for_prompt(meta)

#     return f"""
# You are labeling a software repository.

# Evidence rules:
# - For actor_unit, you MAY use the "Repo metadata" block + the README.
# - For supply_chain_role, research_role, distribution_pathway, and distribution_details, use ONLY the README.

# Task:
# Return exactly one label for each dimension:
# 1) actor_unit (who can change it / enforce controls)
# 2) supply_chain_role (where it sits in the research software supply chain)
# 3) research_role (how directly it contributes to producing research results)
# 4) distribution_pathway (how it is delivered to downstream users)

# Allowed actor_unit values:
# {json.dumps(ACTOR_UNIT_VALUES, indent=2)}

# Allowed supply_chain_role values:
# {json.dumps(SUPPLY_CHAIN_ROLE_VALUES, indent=2)}

# Allowed research_role values:
# {json.dumps(RESEARCH_ROLE_VALUES, indent=2)}

# Allowed distribution_pathway values:
# {json.dumps(DISTRIBUTION_PATHWAY_VALUES, indent=2)}

# Decision rules (follow strictly):

# ACTOR UNIT:
# - Individual maintainer: personal-maintained, no institutional/community governance signs.
# - Research group or lab: tied to a specific lab/project team/PI/students; lab identity present.
# - Institution: university/national lab/institute/government org appears responsible/hosting/operating.
# - Community or foundation: broader open-source governance beyond one lab/institution; formal processes/roles.
#   IMPORTANT: Only choose this if README explicitly indicates community/foundation governance (steering committee, governance model, foundation name, etc).
# - Vendor or commercial entity: company primarily develops/operates it; commercial product/service signals.
# - Platform operator: primary is operating distribution/hosting infra for many projects (registry/CI/hosting).
# - Mixed or shared responsibility: explicit shared responsibility across actor units.
# - Unknown: not enough evidence.

# SUPPLY CHAIN ROLE:
# - Application software: end-user program to run for research tasks.
# - Dependency software artifact: library/package meant to be imported/depended upon.
# - Build and release software: CI/build/test/package/sign/publish tooling.
# - Infrastructure: provides environment/substrate (orchestration/runtime/platform) where software runs.
# - Runtime instrumentation software: profiling/monitoring/instrumentation/modification at runtime.
# - Assistant layer software: interactive assistance in workflows (chatbots / guided automation).
# - Unknown: insufficient README evidence.

# RESEARCH ROLE (Research coupling):
# - Direct research execution: directly generates/transforms/analyzes scientific data/models; removing it prevents producing core scientific outputs (results/models/figures).
# - Research-support tooling: enables development, reproducibility, deployment, management of workflows; not the core analysis/model itself.
# - Incidental or general-purpose: generic tool broadly used across domains, not research-specific in intent/design and not clearly part of a research toolchain.
# - Unknown: insufficient evidence.

# DISTRIBUTION PATHWAY:
# - Source repo: README emphasizes clone/build/install from repo; no clear formal release/registry as primary channel.
# - Releases: README emphasizes tagged releases / GitHub Releases / tarballs as consumption unit.
# - Package registry: README shows ecosystem install command (pip/conda/cran/maven/npm/etc). If chosen, set distribution_details to the registry name.
# - Containers: README emphasizes docker/OCI images (docker pull/run). If chosen, set distribution_details (e.g., Docker Hub/GHCR).
# - Installer or binary: README emphasizes downloadable binaries/installers/executables.
# - Network service: README indicates hosted web app/API/managed service where users interact over network rather than installing.
# - Unknown: insufficient evidence from README.

# Output requirements:
# - Output must match the schema exactly.
# - Provide up to 3 short evidence bullets (quotes or paraphrases) drawn from README.
# - Set confidence between 0 and 1.
# - If uncertain, choose "Unknown" and set low confidence.
# - For distribution_pathway="Package registry" or "Containers", set distribution_details (e.g., "PyPI", "conda-forge", "CRAN", "Maven", "npm", "Docker Hub", "GHCR"). Otherwise distribution_details should be null.

# Repo metadata (allowed evidence ONLY for actor_unit):
# {json.dumps(ctx, indent=2)}

# README (verbatim):
# ----------------
# {readme[:120000]}
# ----------------
# """.strip()


# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow the labeling rules exactly and keep labels within the allowed value sets.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Post-checks / overrides (high ROI)
# # ----------------------------
# _GOVERNANCE_HINTS = re.compile(
#     r"(governance|steering committee|technical steering|foundation|community-led|maintainers?\s+team|"
#     r"contributor covenant|code of conduct|working group|election|rfc|roadmap)",
#     re.IGNORECASE,
# )

# def actor_unit_sanity_override(
#     predicted_actor_unit: str,
#     meta: dict,
#     readme: str,
# ) -> str:
#     """
#     Prevent common hallucination:
#     - If owner_type is User and README doesn't show explicit governance, do NOT label as Community/foundation.
#     """
#     if predicted_actor_unit != "Community or foundation (open source governance)":
#         return predicted_actor_unit

#     data = meta.get("data", {}) if isinstance(meta.get("data"), dict) else {}
#     owner = data.get("owner", {}) if isinstance(data.get("owner"), dict) else {}
#     owner_type = owner.get("type")

#     has_governance = bool(_GOVERNANCE_HINTS.search(readme or ""))

#     if owner_type == "User" and not has_governance:
#         # Conservative: usually an individual maintainer rather than "community".
#         return "Individual maintainer"

#     return predicted_actor_unit


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     prompt = build_prompt(readme, meta)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     # 1) Actor-unit sanity override using repo metadata + governance hints
#     actor_unit_final = actor_unit_sanity_override(result.actor_unit, meta, readme)

#     # 2) Deterministic distribution override (reduces flip-flopping)
#     forced_pathway, forced_details, forced_ev = detect_distribution(readme)
#     distribution_pathway_final = result.distribution_pathway
#     distribution_details_final = result.distribution_details
#     evidence_final = list(result.evidence[:3])

#     if forced_pathway:
#         distribution_pathway_final = forced_pathway
#         distribution_details_final = forced_details
#         # If we override, we can optionally inject a tiny detector evidence line (without exceeding 3 total)
#         for s in forced_ev:
#             if len(evidence_final) >= 3:
#                 break
#             if s not in evidence_final:
#                 evidence_final.append(s)

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": actor_unit_final,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": result.research_role,
#         "distribution_pathway": distribution_pathway_final,
#         "distribution_details": distribution_details_final,
#         "evidence": evidence_final[:3],
#         "confidence": float(result.confidence),
#         "model": model,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     metadata_path.write_text(
#         json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#         encoding="utf-8",
#     )

#     time.sleep(delay)
#     return True, "updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo (default: current directory).")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name (try gpt-5.1 and a mini model).")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing New_SSC_Taxonomy field.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Print a checkpoint log every N processed files.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (repo root) or export it in your shell.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     providers: List[str]
#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import List, Optional, Literal, Tuple

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# # Always load .env from repo root and override any existing shell env var
# # Run this script from the rseng/software repo root so ".env" is found.
# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Build and release software",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]

# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Build and release software",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="0 to 1 confidence score in the labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Prefer human-facing GitHub URL (not api.github.com).
#     """
#     url = metadata.get("url")
#     if isinstance(url, str) and url.strip() and "api.github.com" not in url:
#         return url.strip()

#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         # Prefer html_url first
#         html = data.get("html_url")
#         if isinstance(html, str) and html.strip():
#             return html.strip()

#         # Fall back to clone_url (strip .git)
#         clone = data.get("clone_url")
#         if isinstance(clone, str) and "github.com" in clone and clone.strip():
#             clone = clone.strip()
#             if clone.endswith(".git"):
#                 clone = clone[:-4]
#             if "api.github.com" not in clone:
#                 return clone

#         # Last resort: data["url"] but only if it's not api.github.com
#         v = data.get("url")
#         if isinstance(v, str) and v.strip() and "github.com" in v and "api.github.com" not in v:
#             return v.strip()

#     return None


# def extract_repo_context_for_prompt(metadata: dict) -> dict:
#     """
#     Minimal repo metadata that improves actor_unit classification.
#     Uses only what's already in metadata.json (no GitHub token needed).
#     """
#     data = metadata.get("data", {})
#     owner = {}
#     if isinstance(data, dict):
#         owner = data.get("owner") or {}

#     def _safe_str(x):
#         return x if isinstance(x, str) else None

#     ctx = {
#         "uid": _safe_str(metadata.get("uid")),
#         "parser": _safe_str(metadata.get("parser")),
#         "html_url": _safe_str(data.get("html_url")) if isinstance(data, dict) else None,
#         "full_name": _safe_str(data.get("full_name")) if isinstance(data, dict) else None,
#         "name": _safe_str(data.get("name")) if isinstance(data, dict) else None,
#         "description": _safe_str(data.get("description")) if isinstance(data, dict) else None,
#         "homepage": _safe_str(data.get("homepage")) if isinstance(data, dict) else None,
#         "topics": data.get("topics") if isinstance(data, dict) else None,
#         "owner_login": _safe_str(owner.get("login")) if isinstance(owner, dict) else None,
#         "owner_type": _safe_str(owner.get("type")) if isinstance(owner, dict) else None,  # "User" or "Organization"
#         "owner_html_url": _safe_str(owner.get("html_url")) if isinstance(owner, dict) else None,
#     }

#     # drop empties so prompt stays small
#     return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}


# # ----------------------------
# # README fetching (no GitHub token)
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations via raw.githubusercontent.com.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#         "DOCS.md",
#         "docs/index.md",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if not repo_url or "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Distribution pathway deterministic detector (README-based)
# # Returns README-grounded evidence snippets (quotes/paraphrases referencing actual lines)
# # ----------------------------
# def _first_matching_line(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
#     rx = re.compile(pattern, flags)
#     for line in text.splitlines():
#         if rx.search(line):
#             line = line.strip()
#             if line:
#                 return line[:200]
#     return None


# def detect_distribution_from_readme(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
#     """
#     Returns (pathway, details, evidence_snippets).
#     Evidence snippets must be auditable from README (include the triggering line).
#     """
#     evidence: List[str] = []
#     if not readme:
#         return None, None, evidence

#     # Package registries
#     line = _first_matching_line(readme, r'(^|\s)pip\s+install\s+')
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "PyPI", evidence

#     line = _first_matching_line(readme, r'(^|\s)conda\s+install\s+')
#     if line:
#         details = "conda-forge" if "conda-forge" in line.lower() else "conda"
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", details, evidence

#     # R / CRAN or remotes
#     line = _first_matching_line(readme, r'install\.packages\(|remotes::install_')
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "CRAN (or R ecosystem)", evidence

#     # Julia
#     line = _first_matching_line(readme, r'Pkg\.add\(')
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "Julia General registry", evidence

#     # npm / yarn
#     line = _first_matching_line(readme, r'(^|\s)(npm\s+install|yarn\s+add)\s+')
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "npm", evidence

#     # Maven / Gradle
#     line = _first_matching_line(readme, r'(<dependency>|implementation\s+["\']|mvn\s+)')
#     if line:
#         evidence.append(f'README shows dependency/install info: "{line}"')
#         return "Package registry", "Maven/Gradle ecosystem", evidence

#     # Containers
#     line = _first_matching_line(readme, r'(^|\s)docker\s+(pull|run)\s+')
#     if line:
#         evidence.append(f'README container command: "{line}"')
#         details = "GHCR" if "ghcr.io" in line.lower() else ("Docker Hub" if "docker.io" in line.lower() else None)
#         return "Containers", details, evidence

#     # Releases
#     line = _first_matching_line(readme, r'GitHub\s+Releases|releases\s+page|download\s+the\s+latest\s+release')
#     if line:
#         evidence.append(f'README mentions releases: "{line}"')
#         return "Releases", "GitHub Releases", evidence

#     # Network service
#     line = _first_matching_line(readme, r'(https?://\S+).*(api|endpoint)|REST\s+API|GraphQL|hosted\s+service|web\s+app')
#     if line:
#         evidence.append(f'README indicates network usage: "{line}"')
#         return "Network service", None, evidence

#     # Installer/binary
#     line = _first_matching_line(readme, r'\b(download|installer|binary|exe|dmg|msi)\b')
#     if line:
#         evidence.append(f'README mentions binaries/installers: "{line}"')
#         return "Installer or binary", None, evidence

#     return None, None, evidence


# # ----------------------------
# # Prompting / GPT call
# # - actor_unit can use repo metadata + README
# # - other labels use README only
# # ----------------------------
# def build_prompt(readme: str, repo_metadata: dict) -> str:
#     """
#     Build prompt using PhD student's taxonomy definitions.
#     README is used for classes 2–4.
#     README + metadata (owner.type etc.) may be used for class 1.
#     """

#     owner = repo_metadata.get("data", {}).get("owner", {}) if isinstance(repo_metadata.get("data"), dict) else {}
#     owner_type = owner.get("type")
#     owner_login = owner.get("login")

#     return f"""
# You are labeling a research software artifact using a formal taxonomy.

# You must return exactly one label for each of the four classes:
# 1) actor_unit
# 2) supply_chain_role
# 3) research_role
# 4) distribution_pathway

# Use ONLY README for classes 2, 3, and 4.
# For actor_unit (class 1), you may use README + the provided metadata.

# Provided repo metadata:
# - owner_type: {owner_type}
# - owner_login: {owner_login}

# ===========================
# CLASS 1: ACTOR UNIT
# What it captures:
# The primary organizational unit responsible for producing, maintaining, or operating the research software or its surrounding infrastructure. This dimension helps connect security recommendations to who can realistically implement controls.

# Values and definitions:

# • Individual maintainer  
# Definition: A single person (or an informal, very small group without formal governance) is the primary developer or maintainer.  
# Decision rule: Use when the artifact appears personally maintained, with no clear institutional or community structure.

# • Research group or lab  
# Definition: A research lab or project team is the primary producer/maintainer, often centered on a PI, students, or staff within a lab.  
# Decision rule: Use when maintenance is tied to a specific research group, grant project, or lab identity.

# • Institution (university, lab, government research organization)  
# Definition: A formal institution is responsible for development or operation.  
# Decision rule: Use when metadata or README indicates institutional ownership, institutional hosting, or formal institutional operational responsibility.

# • Community or foundation (open source governance)  
# Definition: A broader open source community or foundation provides governance or stewardship.  
# Decision rule: Use when the artifact is maintained under community governance beyond a single lab or institution, especially with formal processes and roles.

# • Vendor or commercial entity  
# Definition: A company is the primary producer or operator.  
# Decision rule: Use when the artifact is primarily developed or operated by a commercial entity.

# • Platform operator (registry or hosting operator)  
# Definition: An entity whose primary role is operating distribution or hosting infrastructure used by many projects.  
# Decision rule: Use when the security-relevant actor is the operator of the platform rather than the project maintainer.

# • Mixed or shared responsibility  
# Definition: Responsibility is clearly shared across multiple actor units.  
# Decision rule: Use when multiple actor units are explicitly implicated.

# • Unknown  
# Definition: The paper or README does not provide enough information.

# Notes:
# This is not "who uses it." It is "who can change it or enforce controls."
# If owner_type == "User" and no governance structure is described, avoid Community/Foundation.

# ===========================
# CLASS 2: SUPPLY CHAIN ROLE (README only)

# • Application software  
# Definition: A research-facing application that users run to perform a research task.  
# Decision rule: If primarily executed as an end-user program rather than imported, label Application.

# • Dependency software artifact  
# Definition: A reusable library, package, or module intended to be consumed by other software.  
# Decision rule: If imported/linked/depended upon, label Dependency.

# • Build and release software  
# Definition: Tooling that creates artifacts and releases (CI, build scripts, packaging, signing, publishing).  
# Decision rule: If main purpose is building/testing/packaging/publishing software, label Build and release.

# • Infrastructure  
# Definition: Foundational systems that support execution or deployment environments.  
# Decision rule: If it provides environment/substrate on which research software runs, label Infrastructure.

# • Runtime instrumentation software  
# Definition: Tools that observe, instrument, profile, or modify behavior at runtime.  
# Decision rule: If primary role is runtime observation/modification, label Runtime instrumentation.

# • Assistant layer software  
# Definition: Interactive assistance tooling embedded in workflows.  
# Decision rule: If it provides guidance/automation rather than producing artifacts, label Assistant layer.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# CLASS 3: RESEARCH ROLE (README only)

# • Direct research execution  
# Definition: Software that directly generates/transforms/analyzes scientific data/models contributing to research findings.  
# Decision rule: If removing the software would prevent producing core scientific outputs, label Direct.

# • Research-support tooling  
# Definition: Software that supports producing research results but is not itself the core analysis/model.  
# Decision rule: If it enables development, reproducibility, deployment, or workflow management, label Support.

# • Incidental or general-purpose  
# Definition: General software used in research settings but not research-oriented in intent/design.  
# Decision rule: If broadly general-purpose and not research-specific, label Incidental.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# CLASS 4: DISTRIBUTION PATHWAY (README only)

# • Source repo  
# Definition: Primarily distributed by source code repository, users build/install from source.  
# Decision rule: If README emphasizes cloning/building/installing from repo without formal release channel.

# • Releases  
# Definition: Distributed via tagged releases and published artifacts.  
# Decision rule: If release artifacts are the main consumption unit.

# • Package registry  
# Definition: Distributed through registry (CRAN, PyPI, Maven, npm, etc.).  
# Decision rule: If installation is a single ecosystem install command.  
# If chosen, set distribution_details (e.g., "PyPI", "conda-forge").

# • Containers  
# Definition: Distributed as container images (Docker/OCI).  
# Decision rule: If primary unit is a container image.  
# If chosen, set distribution_details (e.g., Docker Hub, GHCR).

# • Installer or binary  
# Definition: Distributed as compiled binaries/installers/executables.  
# Decision rule: If explicitly distinguishes install from binary vs source.

# • Network service  
# Definition: Delivered as hosted or networked service (web app/API/managed platform).  
# Decision rule: If users interact over the network rather than installing.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# Output requirements:
# - Output must match the schema exactly.
# - Provide up to 3 evidence bullets grounded in README text.
# - Set confidence between 0 and 1.
# - If uncertain, choose Unknown and use low confidence.
# - If distribution_pathway is Package registry or Containers, include distribution_details. Otherwise null.

# ===========================
# README:
# {readme[:120000]}
# """.strip()


# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow labeling rules exactly. Do not invent evidence.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     repo_ctx = extract_repo_context_for_prompt(meta)
#     prompt = build_prompt(readme, repo_ctx)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     # Deterministic distribution override (README-only), but keep evidence reviewer-clean
#     override_path, override_details, override_evidence = detect_distribution_from_readme(readme)

#     distribution_pathway = result.distribution_pathway
#     distribution_details = result.distribution_details
#     evidence = list(result.evidence or [])

#     if override_path is not None:
#         distribution_pathway = override_path
#         distribution_details = override_details
#         # Put README-grounded detector evidence first, then model evidence.
#         evidence = (override_evidence + evidence)[:3]
#     else:
#         evidence = evidence[:3]

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": result.actor_unit,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": result.research_role,
#         "distribution_pathway": distribution_pathway,
#         "distribution_details": distribution_details,
#         "evidence": evidence,
#         "confidence": float(result.confidence),
#         "model": model,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     metadata_path.write_text(
#         json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#         encoding="utf-8",
#     )

#     time.sleep(delay)
#     return True, "updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo (default: current directory).")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name (try gpt-5.1 and a mini model).")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing New_SSC_Taxonomy field.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Print a checkpoint log every N processed files.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (repo root) or export it in your shell.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import List, Optional, Literal, Tuple

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# # Always load .env from repo root and override any existing shell env var
# # Run this script from the rseng/software repo root so ".env" is found.
# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Build and release software",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Governance software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]

# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Build and release software",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Governance software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="0 to 1 confidence score in the labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Prefer human-facing GitHub URL (not api.github.com).
#     """
#     url = metadata.get("url")
#     if isinstance(url, str) and url.strip() and "api.github.com" not in url:
#         return url.strip()

#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         # Prefer html_url first
#         html = data.get("html_url")
#         if isinstance(html, str) and html.strip():
#             return html.strip()

#         # Fall back to clone_url (strip .git)
#         clone = data.get("clone_url")
#         if isinstance(clone, str) and "github.com" in clone and clone.strip():
#             clone = clone.strip()
#             if clone.endswith(".git"):
#                 clone = clone[:-4]
#             if "api.github.com" not in clone:
#                 return clone

#         # Last resort: data["url"] but only if it's not api.github.com
#         v = data.get("url")
#         if isinstance(v, str) and v.strip() and "github.com" in v and "api.github.com" not in v:
#             return v.strip()

#     return None


# def extract_repo_context_for_prompt(metadata: dict) -> dict:
#     """
#     Minimal repo metadata that improves actor_unit classification.
#     Uses only what's already in metadata.json (no GitHub token needed).
#     """
#     data = metadata.get("data", {})
#     owner = {}
#     if isinstance(data, dict):
#         owner = data.get("owner") or {}

#     def _safe_str(x):
#         return x if isinstance(x, str) else None

#     ctx = {
#         "uid": _safe_str(metadata.get("uid")),
#         "parser": _safe_str(metadata.get("parser")),
#         "html_url": _safe_str(data.get("html_url")) if isinstance(data, dict) else None,
#         "full_name": _safe_str(data.get("full_name")) if isinstance(data, dict) else None,
#         "name": _safe_str(data.get("name")) if isinstance(data, dict) else None,
#         "description": _safe_str(data.get("description")) if isinstance(data, dict) else None,
#         "homepage": _safe_str(data.get("homepage")) if isinstance(data, dict) else None,
#         "topics": data.get("topics") if isinstance(data, dict) else None,
#         "owner_login": _safe_str(owner.get("login")) if isinstance(owner, dict) else None,
#         "owner_type": _safe_str(owner.get("type")) if isinstance(owner, dict) else None,  # "User" or "Organization"
#         "owner_html_url": _safe_str(owner.get("html_url")) if isinstance(owner, dict) else None,
#     }

#     # drop empties so prompt stays small
#     return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}


# # ----------------------------
# # README fetching (no GitHub token)
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations via raw.githubusercontent.com.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#         "DOCS.md",
#         "docs/index.md",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if not repo_url or "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Distribution pathway deterministic detector (README-based)
# # Returns README-grounded evidence snippets (quotes/paraphrases referencing actual lines)
# # ----------------------------
# def _first_matching_line(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
#     rx = re.compile(pattern, flags)
#     for line in text.splitlines():
#         if rx.search(line):
#             line = line.strip()
#             if line:
#                 return line[:200]
#     return None


# def detect_distribution_from_readme(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
#     """
#     Returns (pathway, details, evidence_snippets).
#     Evidence snippets must be auditable from README (include the triggering line).
#     """
#     evidence: List[str] = []
#     if not readme:
#         return None, None, evidence

#     # Package registries
#     line = _first_matching_line(readme, r"(^|\s)pip\s+install\s+")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "PyPI", evidence

#     line = _first_matching_line(readme, r"(^|\s)conda\s+install\s+")
#     if line:
#         details = "conda-forge" if "conda-forge" in line.lower() else "conda"
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", details, evidence

#     # R / CRAN or remotes
#     line = _first_matching_line(readme, r"install\.packages\(|remotes::install_")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "CRAN (or R ecosystem)", evidence

#     # Julia
#     line = _first_matching_line(readme, r"Pkg\.add\(")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "Julia General registry", evidence

#     # npm / yarn
#     line = _first_matching_line(readme, r"(^|\s)(npm\s+install|yarn\s+add)\s+")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"'.replace("ReADME", "README"))
#         return "Package registry", "npm", evidence

#     # Maven / Gradle
#     line = _first_matching_line(readme, r'(<dependency>|implementation\s+["\']|mvn\s+)')
#     if line:
#         evidence.append(f'README shows dependency/install info: "{line}"')
#         return "Package registry", "Maven/Gradle ecosystem", evidence

#     # Containers
#     line = _first_matching_line(readme, r"(^|\s)docker\s+(pull|run)\s+")
#     if line:
#         evidence.append(f'README container command: "{line}"')
#         details = "GHCR" if "ghcr.io" in line.lower() else ("Docker Hub" if "docker.io" in line.lower() else None)
#         return "Containers", details, evidence

#     # Releases
#     line = _first_matching_line(readme, r"GitHub\s+Releases|releases\s+page|download\s+the\s+latest\s+release")
#     if line:
#         evidence.append(f'README mentions releases: "{line}"')
#         return "Releases", "GitHub Releases", evidence

#     # Network service
#     line = _first_matching_line(readme, r"(https?://\S+).*(api|endpoint)|REST\s+API|GraphQL|hosted\s+service|web\s+app")
#     if line:
#         evidence.append(f'README indicates network usage: "{line}"')
#         return "Network service", None, evidence

#     # Installer/binary
#     line = _first_matching_line(readme, r"\b(download|installer|binary|exe|dmg|msi)\b")
#     if line:
#         evidence.append(f'README mentions binaries/installers: "{line}"')
#         return "Installer or binary", None, evidence

#     return None, None, evidence


# # ----------------------------
# # Post-check overrides (deterministic)
# # ----------------------------

# # Governance signals must be STRONG to justify "Community or foundation"
# _STRONG_GOVERNANCE_HINTS = re.compile(
#     r"(\bgovernance\b|steering committee|technical steering committee|\btsc\b|"
#     r"\bfoundation\b|linux foundation|apache software foundation|cncf|eclipse foundation|"
#     r"\bcharter\b|working group|rfc process|request for comments|"
#     r"maintainers\.md|governance\.md|/governance|/maintainers|"
#     r"code of conduct|contributor covenant)",
#     re.IGNORECASE,
# )

# # Library-ish cues: "library/framework/toolkit/package" etc.
# _LIBRARY_CUES = re.compile(
#     r"(\blibrary\b|\bframework\b|\btoolkit\b|\bpackage\b|\bmodule\b|\bapi\b|"
#     r"\be-graph\b|\bterm rewriting\b|\bmetaprogramming\b|\bcompiler\b|\boptimization\b)",
#     re.IGNORECASE,
# )

# # Direct scientific output cues: produces scientific outputs from data/models
# _DIRECT_OUTPUT_CUES = re.compile(
#     r"(\bfit\b|\bfitting\b|\bcalibrat(e|ed|ion)\b|\bspectrum\b|\btime series\b|\bwaterfall\b|"
#     r"\bresults\b|\bfigures?\b|\bplots?\b|\bexperiment\b|\bobservations?\b|\btelescope\b|"
#     r"\bsimulation\b|\bdata acquisition\b|\bmodel training\b|\btrain\b|\binference\b)",
#     re.IGNORECASE,
# )


# def _get_owner_type(meta: dict) -> Optional[str]:
#     data = meta.get("data", {})
#     if not isinstance(data, dict):
#         return None
#     owner = data.get("owner", {})
#     if not isinstance(owner, dict):
#         return None
#     t = owner.get("type")
#     return t if isinstance(t, str) else None


# def actor_unit_override(actor_unit: str, meta: dict, readme: str) -> str:
#     """
#     Fix common failure:
#     - User-owned repos should NOT be labeled Community/Foundation unless README shows strong governance.
#     """
#     owner_type = _get_owner_type(meta)
#     if owner_type == "User" and actor_unit == "Community or foundation (open source governance)":
#         if not _STRONG_GOVERNANCE_HINTS.search(readme or ""):
#             return "Individual maintainer"
#     return actor_unit


# def research_role_override(research_role: str, supply_chain_role: str, readme: str) -> str:
#     """
#     Fix common failure:
#     - Libraries (Dependency artifacts) often get mislabeled as Direct research execution.
#       Default to Research-support tooling unless README strongly indicates direct research outputs.
#     """
#     if supply_chain_role == "Dependency software artifact":
#         if _LIBRARY_CUES.search(readme or "") and not _DIRECT_OUTPUT_CUES.search(readme or ""):
#             return "Research-support tooling"
#     return research_role


# # ----------------------------
# # Prompting / GPT call
# # - actor_unit can use repo metadata + README
# # - other labels use README only
# # ----------------------------
# def build_prompt(readme: str, repo_metadata: dict) -> str:
#     """
#     Build prompt using PhD student's taxonomy definitions.
#     README is used for classes 2–4.
#     README + metadata (owner.type etc.) may be used for class 1.
#     """

#     owner = repo_metadata.get("data", {}).get("owner", {}) if isinstance(repo_metadata.get("data"), dict) else {}
#     owner_type = owner.get("type")
#     owner_login = owner.get("login")

#     return f"""
# You are labeling a research software artifact using a formal taxonomy.

# You must return exactly one label for each of the four classes:
# 1) actor_unit
# 2) supply_chain_role
# 3) research_role
# 4) distribution_pathway

# Use ONLY README for classes 2, 3, and 4.
# For actor_unit (class 1), you may use README + the provided metadata.

# Provided repo metadata:
# - owner_type: {owner_type}
# - owner_login: {owner_login}

# ===========================
# CLASS 1: ACTOR UNIT
# What it captures:
# The primary organizational unit responsible for producing, maintaining, or operating the research software or its surrounding infrastructure. This dimension helps connect security recommendations to who can realistically implement controls.

# Values and definitions:

# • Individual maintainer  
# Definition: A single person (or an informal, very small group without formal governance) is the primary developer or maintainer.  
# Decision rule: Use when the artifact appears personally maintained, with no clear institutional or community structure.

# • Research group or lab   
# Definition: A research lab or project team is the primary producer/maintainer, often centered on a PI, students, or staff within a lab.  
# Decision rule: Use when maintenance is tied to a specific research group, grant project, or lab identity.

# • Institution (university, lab, government research organization)  
# Definition: A formal institution is responsible for development or operation.  
# Decision rule: Use when metadata or README indicates institutional ownership, institutional hosting, or formal institutional operational responsibility.

# • Community or foundation (open source governance)  
# Definition: A broader open source community or foundation provides governance or stewardship.  
# Decision rule: Use when the artifact is maintained under community governance beyond a single lab or institution, especially with formal processes and roles.

# • Vendor or commercial entity  
# Definition: A company is the primary producer or operator.  
# Decision rule: Use when the artifact is primarily developed or operated by a commercial entity.

# • Platform operator (registry or hosting operator)  
# Definition: An entity whose primary role is operating distribution or hosting infrastructure used by many projects.  
# Decision rule: Use when the security-relevant actor is the operator of the platform rather than the project maintainer.

# • Mixed or shared responsibility  
# Definition: Responsibility is clearly shared across multiple actor units.  
# Decision rule: Use when multiple actor units are explicitly implicated.

# • Unknown  
# Definition: The paper or README does not provide enough information.

# Notes:
# This is not "who uses it." It is "who can change it or enforce controls."
# If owner_type == "User" and no governance structure is described, avoid Community/Foundation.

# ===========================
# CLASS 2: SUPPLY CHAIN ROLE (README only)

# • Application software  
# Definition: A research-facing application that users run to perform a research task.  
# Decision rule: If primarily executed as an end-user program rather than imported, label Application.

# • Dependency software artifact  
# Definition: A reusable library, package, or module intended to be consumed by other software.  
# Decision rule: If imported/linked/depended upon, label Dependency.

# • Build and release software  
# Definition: Tooling that creates artifacts and releases (CI, build scripts, packaging, signing, publishing).  
# Decision rule: If main purpose is building/testing/packaging/publishing software, label Build and release.

# • Infrastructure  
# Definition: Foundational systems that support execution or deployment environments.  
# Decision rule: If it provides environment/substrate on which research software runs, label Infrastructure.

# • Runtime instrumentation software  
# Definition: Tools that observe, instrument, profile, or modify behavior at runtime.  
# Decision rule: If primary role is runtime observation/modification, label Runtime instrumentation.

# • Governance software  
# Definition: Systems and processes that control or gate distribution to users (e.g., package registries, registry policy checks, repository acceptance checks, curated catalogs).  
# Decision rule: If it mediates how artifacts enter an ecosystem or are distributed at scale, label Governance software.

# • Assistant layer software  
# Definition: Interactive assistance tooling embedded in workflows.  
# Decision rule: If it provides guidance/automation rather than producing artifacts, label Assistant layer.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# CLASS 3: RESEARCH ROLE (README only)

# • Direct research execution  
# Definition: Software that directly generates/transforms/analyzes scientific data/models contributing to research findings.  
# Decision rule: If removing the software would prevent producing core scientific outputs, label Direct.

# • Research-support tooling  
# Definition: Software that supports producing research results but is not itself the core analysis/model.  
# Decision rule: If it enables development, reproducibility, deployment, or workflow management, label Support.

# • Incidental or general-purpose  
# Definition: General software used in research settings but not research-oriented in intent/design.  
# Decision rule: If broadly general-purpose and not research-specific, label Incidental.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# CLASS 4: DISTRIBUTION PATHWAY (README only)

# • Source repo  
# Definition: Primarily distributed by source code repository, users build/install from source.  
# Decision rule: If README emphasizes cloning/building/installing from repo without formal release channel.

# • Releases  
# Definition: Distributed via tagged releases and published artifacts.  
# Decision rule: If release artifacts are the main consumption unit.

# • Package registry  
# Definition: Distributed through registry (CRAN, PyPI, Maven, npm, etc.).  
# Decision rule: If installation is a single ecosystem install command.  
# If chosen, set distribution_details (e.g., "PyPI", "conda-forge").

# • Containers  
# Definition: Distributed as container images (Docker/OCI).  
# Decision rule: If primary unit is a container image.  
# If chosen, set distribution_details (e.g., Docker Hub, GHCR).

# • Installer or binary  
# Definition: Distributed as compiled binaries/installers/executables.  
# Decision rule: If explicitly distinguishes install from binary vs source.

# • Network service  
# Definition: Delivered as hosted or networked service (web app/API/managed platform).  
# Decision rule: If users interact over the network rather than installing.

# • Unknown  
# Definition: Insufficient information.

# ===========================
# Output requirements:
# - Output must match the schema exactly.
# - Provide up to 3 evidence bullets grounded in README text.
# - Set confidence between 0 and 1.
# - If uncertain, choose Unknown and use low confidence.
# - If distribution_pathway is Package registry or Containers, include distribution_details. Otherwise null.

# ===========================
# README:
# {readme[:120000]}
# """.strip()


# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow labeling rules exactly. Do not invent evidence.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     repo_ctx = extract_repo_context_for_prompt(meta)
#     prompt = build_prompt(readme, repo_ctx)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     # Deterministic post-check overrides (fixes common failure modes)
#     actor_unit = actor_unit_override(result.actor_unit, meta, readme)
#     research_role = research_role_override(result.research_role, result.supply_chain_role, readme)

#     # Deterministic distribution override (README-only), but keep evidence reviewer-clean
#     override_path, override_details, override_evidence = detect_distribution_from_readme(readme)

#     distribution_pathway = result.distribution_pathway
#     distribution_details = result.distribution_details
#     evidence = list(result.evidence or [])

#     if override_path is not None:
#         distribution_pathway = override_path
#         distribution_details = override_details
#         # Put README-grounded detector evidence first, then model evidence.
#         evidence = (override_evidence + evidence)[:3]
#     else:
#         evidence = evidence[:3]

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": actor_unit,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": research_role,
#         "distribution_pathway": distribution_pathway,
#         "distribution_details": distribution_details,
#         "evidence": evidence,
#         "confidence": float(result.confidence),
#         "model": model,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     metadata_path.write_text(
#         json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#         encoding="utf-8",
#     )

#     time.sleep(delay)
#     return True, "updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo (default: current directory).")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name (try gpt-5.1 and a mini model).")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing New_SSC_Taxonomy field.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Print a checkpoint log every N processed files.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (repo root) or export it in your shell.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()






#!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import List, Optional, Literal, Tuple

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# # Always load .env from repo root and override any existing shell env var
# # Run this script from the rseng/software repo root so ".env" is found.
# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Governance software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Builds and Releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]

# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Governance software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )

#     confidence: float = Field(
#         ge=0.0,
#         le=1.0,
#         description="0 to 1 confidence score in the labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Prefer human-facing GitHub URL (not api.github.com).
#     """
#     url = metadata.get("url")
#     if isinstance(url, str) and url.strip() and "api.github.com" not in url:
#         return url.strip()

#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         # Prefer html_url first
#         html = data.get("html_url")
#         if isinstance(html, str) and html.strip():
#             return html.strip()

#         # Fall back to clone_url (strip .git)
#         clone = data.get("clone_url")
#         if isinstance(clone, str) and "github.com" in clone and clone.strip():
#             clone = clone.strip()
#             if clone.endswith(".git"):
#                 clone = clone[:-4]
#             if "api.github.com" not in clone:
#                 return clone

#         # Last resort: data["url"] but only if it's not api.github.com
#         v = data.get("url")
#         if isinstance(v, str) and v.strip() and "github.com" in v and "api.github.com" not in v:
#             return v.strip()

#     return None


# def extract_repo_context_for_prompt(metadata: dict) -> dict:
#     """
#     Minimal repo metadata that improves actor_unit classification.
#     Uses only what's already in metadata.json (no GitHub token needed).
#     """
#     data = metadata.get("data", {})
#     owner = {}
#     if isinstance(data, dict):
#         owner = data.get("owner") or {}

#     def _safe_str(x):
#         return x if isinstance(x, str) else None

#     ctx = {
#         "uid": _safe_str(metadata.get("uid")),
#         "parser": _safe_str(metadata.get("parser")),
#         "html_url": _safe_str(data.get("html_url")) if isinstance(data, dict) else None,
#         "full_name": _safe_str(data.get("full_name")) if isinstance(data, dict) else None,
#         "name": _safe_str(data.get("name")) if isinstance(data, dict) else None,
#         "description": _safe_str(data.get("description")) if isinstance(data, dict) else None,
#         "homepage": _safe_str(data.get("homepage")) if isinstance(data, dict) else None,
#         "topics": data.get("topics") if isinstance(data, dict) else None,
#         "owner_login": _safe_str(owner.get("login")) if isinstance(owner, dict) else None,
#         "owner_type": _safe_str(owner.get("type")) if isinstance(owner, dict) else None,  # "User" or "Organization"
#         "owner_html_url": _safe_str(owner.get("html_url")) if isinstance(owner, dict) else None,
#     }

#     # drop empties so prompt stays small
#     return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}


# # ----------------------------
# # README fetching (no GitHub token)
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations via raw.githubusercontent.com.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#         "DOCS.md",
#         "docs/index.md",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if not repo_url or "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Distribution pathway deterministic detector (README-based)
# # Returns README-grounded evidence snippets
# # ----------------------------
# def _first_matching_line(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
#     rx = re.compile(pattern, flags)
#     for line in text.splitlines():
#         if rx.search(line):
#             line = line.strip()
#             if line:
#                 return line[:200]
#     return None


# def detect_distribution_from_readme(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
#     """
#     Returns (pathway, details, evidence_snippets).
#     Evidence snippets must be auditable from README (include the triggering line).
#     """
#     evidence: List[str] = []
#     if not readme:
#         return None, None, evidence

#     # Package registries
#     line = _first_matching_line(readme, r"(^|\s)pip\s+install\s+")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"')
#         return "Package registry", "PyPI", evidence

#     line = _first_matching_line(readme, r"(^|\s)conda\s+install\s+")
#     if line:
#         details = "conda-forge" if "conda-forge" in line.lower() else "conda"
#         evidence.append(f'ReADME install command: "{line}"')
#         return "Package registry", details, evidence

#     line = _first_matching_line(readme, r"install\.packages\(|remotes::install_")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"')
#         return "Package registry", "CRAN (or R ecosystem)", evidence

#     line = _first_matching_line(readme, r"Pkg\.add\(")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"')
#         return "Package registry", "Julia General registry", evidence

#     line = _first_matching_line(readme, r"(^|\s)(npm\s+install|yarn\s+add)\s+")
#     if line:
#         evidence.append(f'ReADME install command: "{line}"')
#         return "Package registry", "npm", evidence

#     line = _first_matching_line(readme, r'(<dependency>|implementation\s+["\']|mvn\s+)')
#     if line:
#         evidence.append(f'ReADME shows dependency/install info: "{line}"')
#         return "Package registry", "Maven/Gradle ecosystem", evidence

#     # Containers
#     line = _first_matching_line(readme, r"(^|\s)docker\s+(pull|run)\s+")
#     if line:
#         evidence.append(f'ReADME container command: "{line}"')
#         details = "GHCR" if "ghcr.io" in line.lower() else ("Docker Hub" if "docker.io" in line.lower() else None)
#         return "Containers", details, evidence

#     # Releases
#     line = _first_matching_line(readme, r"GitHub\s+Releases|releases\s+page|download\s+the\s+latest\s+release")
#     if line:
#         evidence.append(f'ReADME mentions releases: "{line}"')
#         return "Releases", "GitHub Releases", evidence

#     # Network service
#     line = _first_matching_line(readme, r"(https?://\S+).*(api|endpoint)|REST\s+API|GraphQL|hosted\s+service|web\s+app")
#     if line:
#         evidence.append(f'ReADME indicates network usage: "{line}"')
#         return "Network service", None, evidence

#     # Installer/binary
#     line = _first_matching_line(readme, r"\b(download|installer|binary|exe|dmg|msi)\b")
#     if line:
#         evidence.append(f'ReADME mentions binaries/installers: "{line}"')
#         return "Installer or binary", None, evidence

#     return None, None, evidence


# # ----------------------------
# # Post-check overrides (deterministic)
# # ----------------------------
# _STRONG_GOVERNANCE_HINTS = re.compile(
#     r"(\bgovernance\b|steering committee|technical steering committee|\btsc\b|"
#     r"\bfoundation\b|linux foundation|apache software foundation|cncf|eclipse foundation|"
#     r"\bcharter\b|working group|rfc process|request for comments|"
#     r"maintainers\.md|governance\.md|/governance|/maintainers|"
#     r"code of conduct|contributor covenant)",
#     re.IGNORECASE,
# )

# _INSTITUTIONAL_HINTS = re.compile(
#     r"(\buniversity\b|\bcollege\b|\binstitute\b|\.edu\b|"
#     r"\bnational lab\b|\bresearch lab\b|\bresearch center\b|"
#     r"\bnsf\b|\bnih\b|\bdoe\b|\bnasa\b|grant|funded by|"
#     r"\bacademic\b|\bdepartment of\b)",
#     re.IGNORECASE,
# )

# _COMMERCIAL_HINTS = re.compile(
#     r"(\bcorp\b|\bcorporation\b|\bcompany\b|\bltd\b|\binc\b|\bllc\b|"
#     r"\benterprise\b|\bcommercial\b|\bproduct\b|\bsolutions\b|"
#     r"\bcopyright.*(?:corp|inc|ltd|llc)\b)",
#     re.IGNORECASE,
# )

# _RESEARCH_GROUP_HINTS = re.compile(
#     r"(\blab\b|\bgroup\b|\bresearch group\b|\bresearch team\b|"
#     r"\bPI\b|\bprincipal investigator\b|\bpostdoc\b|\bphd student\b|"
#     r"\bgraduate student\b|\bundergraduate\b)",
#     re.IGNORECASE,
# )

# _LIBRARY_CUES = re.compile(
#     r"(\blibrary\b|\bframework\b|\btoolkit\b|\bpackage\b|\bmodule\b|\bapi\b|"
#     r"\bsdk\b|\bplugin\b|\bextension\b)",
#     re.IGNORECASE,
# )

# _DIRECT_OUTPUT_CUES = re.compile(
#     r"(\bresults\b|\bfigures?\b|\bplots?\b|\bexperiment\b|\bobservations?\b|"
#     r"\bsimulation\b|\banalysis\b|\bvisuali[zs]ation\b|\bdata processing\b|\bpipeline\b|"
#     r"\bspectrum\b|\btime series\b|\btelescope\b|\bmodel\b|\btrain\b|\binference\b)",
#     re.IGNORECASE,
# )

# _APPLICATION_CUES = re.compile(
#     r"(\bcommand.?line\b|\bcli\b|\bgui\b|\bprogram\b|\bapplication\b|"
#     r"\brun\b|\bexecute\b|\blaunch\b)",
#     re.IGNORECASE,
# )


# def _get_owner_type(meta: dict) -> Optional[str]:
#     data = meta.get("data", {})
#     if not isinstance(data, dict):
#         return None
#     owner = data.get("owner", {})
#     if not isinstance(owner, dict):
#         return None
#     t = owner.get("type")
#     return t if isinstance(t, str) else None


# def _get_owner_login(meta: dict) -> Optional[str]:
#     data = meta.get("data", {})
#     if not isinstance(data, dict):
#         return None
#     owner = data.get("owner", {})
#     if not isinstance(owner, dict):
#         return None
#     login = owner.get("login")
#     return login if isinstance(login, str) else None


# def actor_unit_override(actor_unit: str, meta: dict, readme: str) -> str:
#     """
#     Actor unit sanity checks using metadata.json + README.
#     """
#     owner_type = _get_owner_type(meta)
#     readme_text = readme or ""

#     # User-owned repos should NOT be Community/Foundation without strong governance
#     if owner_type == "User" and actor_unit == "Community or foundation (open source governance)":
#         if not _STRONG_GOVERNANCE_HINTS.search(readme_text):
#             return "Individual maintainer"

#     # If labeled Individual maintainer but README screams institutional, bump
#     if actor_unit == "Individual maintainer" and _INSTITUTIONAL_HINTS.search(readme_text):
#         if _RESEARCH_GROUP_HINTS.search(readme_text):
#             return "Research group or lab"
#         return "Institution (university, lab, government research organization)"

#     # If labeled Individual maintainer but commercial signals appear
#     if actor_unit == "Individual maintainer" and _COMMERCIAL_HINTS.search(readme_text):
#         return "Vendor or commercial entity"

#     # If Unknown and org-owned, try to infer
#     if owner_type == "Organization" and actor_unit == "Unknown":
#         if _RESEARCH_GROUP_HINTS.search(readme_text):
#             return "Research group or lab"
#         if _INSTITUTIONAL_HINTS.search(readme_text):
#             return "Institution (university, lab, government research organization)"
#         if _COMMERCIAL_HINTS.search(readme_text):
#             return "Vendor or commercial entity"

#     return actor_unit


# def research_role_override(research_role: str, supply_chain_role: str, readme: str) -> str:
#     """
#     Conservative guardrails for research_role based on README cues + supply_chain_role.
#     """
#     readme_text = readme or ""

#     # Build/Infrastructure/Runtime are almost always support
#     if supply_chain_role in [
#         "Build and release software",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Governance software",
#     ]:
#         if research_role == "Direct research execution":
#             return "Research-support tooling"

#     # Dependencies tend to be support unless clearly producing scientific outputs
#     if supply_chain_role == "Dependency software artifact":
#         has_lib = bool(_LIBRARY_CUES.search(readme_text))
#         has_direct = bool(_DIRECT_OUTPUT_CUES.search(readme_text))
#         if has_lib and not has_direct:
#             return "Research-support tooling"

#         # If it's strongly described as a library/toolkit, lean support
#         app_matches = len(_APPLICATION_CUES.findall(readme_text))
#         lib_matches = len(_LIBRARY_CUES.findall(readme_text))
#         if lib_matches > app_matches * 2:
#             return "Research-support tooling"

#     # Applications with direct output cues => direct execution
#     if supply_chain_role == "Application software":
#         if _DIRECT_OUTPUT_CUES.search(readme_text):
#             return "Direct research execution"

#     # If claimed direct but no direct cues, prefer support (conservative)
#     if research_role == "Direct research execution" and not _DIRECT_OUTPUT_CUES.search(readme_text):
#         if _LIBRARY_CUES.search(readme_text):
#             return "Research-support tooling"

#     return research_role


# # ----------------------------
# # Prompting / GPT call
# # ----------------------------
# def build_prompt(readme: str, repo_metadata: dict) -> str:
#     owner_type = repo_metadata.get("owner_type")
#     owner_login = repo_metadata.get("owner_login")

#     return f"""
# You are labeling a research software artifact using a formal taxonomy.

# You must return exactly one label for each of the four classes:
# 1) actor_unit
# 2) supply_chain_role
# 3) research_role
# 4) distribution_pathway

# Use ONLY README for classes 2, 3, and 4.
# For actor_unit (class 1), you may use README + the provided metadata.

# Provided repo metadata:
# - owner_type: {owner_type}
# - owner_login: {owner_login}

# ===========================
# CLASS 1: Actor unit (who the relevant actor is)
# What it captures:
# The primary organizational unit responsible for producing, maintaining, or operating the research software or its surrounding infrastructure. This dimension helps connect security recommendations to who can realistically implement controls.

# Values and definitions
# • Individual maintainer
# Definition: A single person (or an informal, very small group without formal governance) is the primary developer or maintainer.
# Decision rule: Use when the artifact appears personally maintained, with no clear institutional or community structure.

# • Research group or lab
# Definition: A research lab or project team is the primary producer/maintainer, often centered on a PI, students, or staff within a lab.
# Decision rule: Use when maintenance is tied to a specific research group, grant project, or lab identity.

# • Institution (university, lab, government research organization)
# Definition: A formal institution is responsible for development or operation (for example, university-supported software, national lab software, or institute-managed infrastructure).
# Decision rule: Use when the paper or metadata indicates institutional ownership, institutional hosting, or formal institutional operational responsibility.

# • Community or foundation (open source governance)
# Definition: A broader open source community or foundation provides governance or stewardship (for example, a named community project, foundation-backed ecosystem, or standards body).
# Decision rule: Use when the artifact is maintained under community governance beyond a single lab or institution, especially with formal processes and roles.

# • Vendor or commercial entity
# Definition: A company is the primary producer or operator, including commercial services and proprietary or dual-licensed tools.
# Decision rule: Use when the artifact is primarily developed or operated by a commercial entity.

# • Platform operator (registry or hosting operator)
# Definition: An entity whose primary role is operating distribution or hosting infrastructure used by many projects (for example, package registries, repository hosting, CI platform operators).
# Decision rule: Use when the security-relevant actor is the operator of the distribution or hosting platform rather than a software project maintainer.

# • Mixed or shared responsibility
# Definition: Responsibility is clearly shared across multiple actor units (for example, a lab produces the software but a foundation governs releases, or a community maintains while a vendor operates infrastructure).
# Decision rule: Use when multiple actor units are explicitly implicated.

# • Unknown
# Definition: The paper does not provide enough information to identify the responsible actor unit.

# Notes on how to use it
# This is not “who uses it.” It is “who can change it or enforce controls.”
# If your unit of analysis is a registry policy (like CRAN checks), the actor unit is often Platform operator even if the software packages are maintained by individuals or labs.

# ===========================
# CLASS 2: Supply chain role (README only)
# What it captures:
# Where the artifact sits in the research software supply chain and how a compromise would propagate.

# • Application software
# Definition: A research-facing application that users run to perform a research task (for example, a simulator, analysis program, scientific workflow application).
# Decision rule: If it is primarily executed as an end-user program rather than imported as a dependency, label as Application.

# • Dependency software artifact
# Definition: A reusable library, package, or module intended to be consumed by other software (including transitive dependency components).
# Decision rule: If it is imported, linked, or depended upon by other projects as a component, label as Dependency.

# • Infrastructure
# Definition: Foundational systems that support execution or deployment environments (for example, containerization frameworks, orchestration infrastructure, runtime platforms).
# Decision rule: If it provides an environment or substrate upon which research software runs or is deployed, label as Infrastructure.

# • Runtime instrumentation software
# Definition: Tools that observe, instrument, profile, or modify behavior at runtime for testing, performance, monitoring, or compliance.
# Decision rule: If its primary role is runtime observation or modification rather than producing artifacts or distributing them, label as Runtime instrumentation.

# • Governance software
# Definition: Systems and processes that control or gate distribution to users (for example, package registries, registry policy checks, repository acceptance checks, curated catalogs).
# Decision rule: If it mediates how artifacts enter an ecosystem or are distributed at scale, label as Governance software.

# • Assistant layer software
# Definition: Interactive assistance tooling embedded in developer or community workflows (for example, chat-based assistants integrated into collaboration platforms that guide usage or development).
# Decision rule: If the artifact provides interactive guidance, automation, or decision support within the ecosystem rather than being part of build or distribution, label as Assistant layer.

# • Unknown
# Definition: Insufficient information to determine role.

# ===========================
# CLASS 3: Research coupling or Research role (README only)
# What it captures:
# How directly the software contributes to producing research results versus supporting the research process.

# • Direct research execution
# Definition: Software that directly generates, transforms, or analyzes scientific data or models in a way that contributes to research findings (for example, simulation codes, analysis pipelines, domain tools that produce figures or results).
# Decision rule: If removing the software would prevent producing the core scientific output (results, models, figures), label as Direct.

# • Research-support tooling
# Definition: Software that supports producing research results but is not itself the core analysis or modeling artifact (for example, workflow orchestration, packaging, testing, data management tooling, containerization frameworks, build and release tooling).
# Decision rule: If the software primarily enables development, execution, reproducibility, deployment, or management of research workflows, label as Support.

# • Incidental or general-purpose
# Definition: General software used in research settings but not research-oriented in design or intent (for example, generic OS components, generic editors, general infrastructure unrelated to research goals).
# Decision rule: If the software is broadly used across domains without being research-specific and is not part of a research toolchain intentionally, label as Incidental.

# • Unknown
# Definition: Insufficient information to determine coupling.

# ===========================
# CLASS 4: Distribution pathway (README only)
# What it captures:
# How the software is delivered to downstream users and systems.

# • Source repo
# Definition: Primarily distributed by source code repository, with users building or installing directly from source.
# Decision rule: If the paper emphasizes cloning, building, or installing from a repository without a formal release channel, use Source repo.

# • Builds and releases
# Definition: The software is delivered to downstream users primarily through published build artifacts and releases, such as versioned tags and downloadable assets produced via build and release processes (for example, CI workflows, build scripts, packaging, signing, and publishing). 
# Decision rule: Use Builds and releases when the main unit consumed by users is a published build artifact or release asset, such as a tagged version with attached binaries or archives.

# • Package registry
# Definition: Distributed through a registry or ecosystem package manager (for example, CRAN, PyPI, Maven, npm).
# Decision rule: If installation is described as a single command via an ecosystem registry, use Package registry and specify which.

# • Containers
# Definition: Distributed as container images (for example, Docker or OCI images) or heavily dependent on container-based delivery.
# Decision rule: If the primary operational unit is a container image, use Containers.

# • Installer or binary
# Definition: Distributed as compiled binaries, installers, or downloadable executables.
# Decision rule: If the paper explicitly distinguishes install from binary versus from source, use Installer or binary.

# • Network service
# Definition: Delivered as a hosted or networked service (for example, web apps, APIs, managed platforms).
# Decision rule: If users interact with it over the network rather than installing it, use Network service.

# • Unknown
# Definition: Insufficient information to determine pathway.

# ===========================
# Output requirements:
# - Output must match the schema exactly.
# - Provide up to 3 evidence bullets grounded in README text.
# - Set confidence between 0 and 1.
# - If uncertain, choose Unknown and use low confidence.
# - distribution_details should be null unless pathway is Package registry or Containers.

# ===========================
# README:
# {readme[:120000]}
# """.strip()


# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow the taxonomy rules exactly. Do not invent evidence.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
#     dry_run: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     repo_ctx = extract_repo_context_for_prompt(meta)
#     prompt = build_prompt(readme, repo_ctx)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     # Post-check overrides
#     actor_unit_original = result.actor_unit
#     research_role_original = result.research_role

#     actor_unit = actor_unit_override(result.actor_unit, meta, readme)
#     research_role = research_role_override(result.research_role, result.supply_chain_role, readme)

#     # Deterministic distribution override (README-only)
#     override_path, override_details, override_evidence = detect_distribution_from_readme(readme)

#     distribution_pathway = result.distribution_pathway
#     distribution_details = result.distribution_details
#     evidence = list(result.evidence or [])

#     if override_path is not None:
#         distribution_pathway = override_path
#         distribution_details = override_details
#         # Put README-grounded detector evidence first, then model evidence.
#         evidence = (override_evidence + evidence)[:3]
#     else:
#         evidence = evidence[:3]

#     overrides_applied = {
#         "actor_unit": actor_unit != actor_unit_original,
#         "research_role": research_role != research_role_original,
#         "distribution_pathway": override_path is not None,
#     }

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": actor_unit,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": research_role,
#         "distribution_pathway": distribution_pathway,
#         "distribution_details": distribution_details,
#         "evidence": evidence,
#         "confidence": float(result.confidence),
#         "model": model,
#         "overrides_applied": overrides_applied,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     if not dry_run:
#         metadata_path.write_text(
#             json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#             encoding="utf-8",
#         )

#     time.sleep(delay)
#     return True, "updated" if not dry_run else "dry_run_updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo (default: current directory).")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name (e.g., gpt-5.1).")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing New_SSC_Taxonomy field.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Print a checkpoint log every N processed files.")
#     ap.add_argument("--dry-run", action="store_true", help="Do not write changes to disk.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (repo root) or export it in your shell.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     providers: List[str]
#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")
#     if args.dry_run:
#         print("NOTE: dry-run enabled (no files will be modified).")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#             dry_run=args.dry_run,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# from __future__ import annotations

# import argparse
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import List, Optional, Literal, Tuple

# import requests
# from tqdm import tqdm
# from pydantic import BaseModel, Field
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=".env", override=True)

# # ----------------------------
# # Allowed label values
# # ----------------------------
# ACTOR_UNIT_VALUES = [
#     "Individual maintainer",
#     "Research group or lab",
#     "Institution (university, lab, government research organization)",
#     "Community or foundation (open source governance)",
#     "Vendor or commercial entity",
#     "Platform operator (registry or hosting operator)",
#     "Mixed or shared responsibility",
#     "Unknown",
# ]

# SUPPLY_CHAIN_ROLE_VALUES = [
#     "Application software",
#     "Dependency software artifact",
#     "Infrastructure",
#     "Runtime instrumentation software",
#     "Governance software",
#     "Assistant layer software",
#     "Unknown",
# ]

# RESEARCH_ROLE_VALUES = [
#     "Direct research execution",
#     "Research-support tooling",
#     "Incidental or general-purpose",
#     "Unknown",
# ]

# DISTRIBUTION_PATHWAY_VALUES = [
#     "Source repo",
#     "Builds and releases",
#     "Package registry",
#     "Containers",
#     "Installer or binary",
#     "Network service",
#     "Unknown",
# ]

# # ----------------------------
# # Structured output schema
# # ----------------------------
# class SSCResult(BaseModel):
#     actor_unit: Literal[
#         "Individual maintainer",
#         "Research group or lab",
#         "Institution (university, lab, government research organization)",
#         "Community or foundation (open source governance)",
#         "Vendor or commercial entity",
#         "Platform operator (registry or hosting operator)",
#         "Mixed or shared responsibility",
#         "Unknown",
#     ]

#     supply_chain_role: Literal[
#         "Application software",
#         "Dependency software artifact",
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Governance software",
#         "Assistant layer software",
#         "Unknown",
#     ]

#     research_role: Literal[
#         "Direct research execution",
#         "Research-support tooling",
#         "Incidental or general-purpose",
#         "Unknown",
#     ]

#     distribution_pathway: Literal[
#         "Source repo",
#         "Builds and releases",
#         "Package registry",
#         "Containers",
#         "Installer or binary",
#         "Network service",
#         "Unknown",
#     ]

#     distribution_details: Optional[str] = Field(
#         default=None,
#         description="Optional detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
#     )

#     evidence: List[str] = Field(
#         default_factory=list,
#         description="Up to 3 short quotes/paraphrases from README supporting labels.",
#     )


# # ----------------------------
# # Metadata helpers
# # ----------------------------
# def extract_repo_url(metadata: dict) -> Optional[str]:
#     """
#     Schemas vary. Prefer human-facing GitHub URL (not api.github.com).
#     """
#     url = metadata.get("url")
#     if isinstance(url, str) and url.strip() and "api.github.com" not in url:
#         return url.strip()

#     data = metadata.get("data", {})
#     if isinstance(data, dict):
#         html = data.get("html_url")
#         if isinstance(html, str) and html.strip():
#             return html.strip()

#         clone = data.get("clone_url")
#         if isinstance(clone, str) and "github.com" in clone and clone.strip():
#             clone = clone.strip()
#             if clone.endswith(".git"):
#                 clone = clone[:-4]
#             if "api.github.com" not in clone:
#                 return clone

#         v = data.get("url")
#         if isinstance(v, str) and v.strip() and "github.com" in v and "api.github.com" not in v:
#             return v.strip()

#     return None


# def extract_repo_context_for_prompt(metadata: dict) -> dict:
#     """
#     Minimal repo metadata that improves actor_unit classification.
#     """
#     data = metadata.get("data", {})
#     owner = {}
#     if isinstance(data, dict):
#         owner = data.get("owner") or {}

#     def _safe_str(x):
#         return x if isinstance(x, str) else None

#     ctx = {
#         "uid": _safe_str(metadata.get("uid")),
#         "parser": _safe_str(metadata.get("parser")),
#         "html_url": _safe_str(data.get("html_url")) if isinstance(data, dict) else None,
#         "full_name": _safe_str(data.get("full_name")) if isinstance(data, dict) else None,
#         "name": _safe_str(data.get("name")) if isinstance(data, dict) else None,
#         "description": _safe_str(data.get("description")) if isinstance(data, dict) else None,
#         "homepage": _safe_str(data.get("homepage")) if isinstance(data, dict) else None,
#         "topics": data.get("topics") if isinstance(data, dict) else None,
#         "owner_login": _safe_str(owner.get("login")) if isinstance(owner, dict) else None,
#         "owner_type": _safe_str(owner.get("type")) if isinstance(owner, dict) else None,
#         "owner_html_url": _safe_str(owner.get("html_url")) if isinstance(owner, dict) else None,
#     }

#     return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}


# # ----------------------------
# # README fetching (no GitHub token)
# # ----------------------------
# def github_raw_readme_urls(repo_url: str) -> List[str]:
#     """
#     Try common README locations via raw.githubusercontent.com.
#     """
#     repo_url = repo_url.rstrip("/")
#     parts = repo_url.split("/")
#     if len(parts) < 2:
#         return []

#     owner, repo = parts[-2], parts[-1]
#     base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
#     candidates = [
#         "README.md",
#         "README.MD",
#         "README",
#         "readme.md",
#         "Readme.md",
#         "docs/README.md",
#         "README.rst",
#         "README.txt",
#         "DOCS.md",
#         "docs/index.md",
#     ]
#     return [base + c for c in candidates]


# def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
#     if not repo_url or "github.com" not in repo_url:
#         return None

#     headers = {"User-Agent": "rse-ssc-annotator/1.0"}

#     for raw in github_raw_readme_urls(repo_url):
#         try:
#             r = requests.get(raw, timeout=timeout, headers=headers)
#             if r.status_code == 200 and r.text and r.text.strip():
#                 return r.text
#         except Exception:
#             continue

#     return None


# # ----------------------------
# # Distribution pathway deterministic detector
# # ----------------------------
# def _first_matching_line(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
#     rx = re.compile(pattern, flags)
#     for line in text.splitlines():
#         if rx.search(line):
#             line = line.strip()
#             if line:
#                 return line[:200]
#     return None


# def detect_distribution_from_readme(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
#     """
#     Returns (pathway, details, evidence_snippets).
#     Evidence snippets are auditable from README.
#     """
#     evidence: List[str] = []
#     if not readme:
#         return None, None, evidence

#     # Package registries
#     line = _first_matching_line(readme, r"(^|\s)pip\s+install\s+")
#     if line:
#         evidence.append(f'README install command: "{line}"')
#         return "Package registry", "PyPI", evidence

#     line = _first_matching_line(readme, r"(^|\s)conda\s+install\s+")
#     if line:
#         details = "conda-forge" if "conda-forge" in line.lower() else "conda"
#         evidence.append(f'README install command: "{line}"')
#         return "Package registry", details, evidence

#     line = _first_matching_line(readme, r"install\.packages\(|remotes::install_")
#     if line:
#         evidence.append(f'README install command: "{line}"')
#         return "Package registry", "CRAN (or R ecosystem)", evidence

#     line = _first_matching_line(readme, r"Pkg\.add\(")
#     if line:
#         evidence.append(f'README install command: "{line}"')
#         return "Package registry", "Julia General registry", evidence

#     line = _first_matching_line(readme, r"(^|\s)(npm\s+install|yarn\s+add)\s+")
#     if line:
#         evidence.append(f'README install command: "{line}"')
#         return "Package registry", "npm", evidence

#     line = _first_matching_line(readme, r'(<dependency>|implementation\s+["\']|mvn\s+)')
#     if line:
#         evidence.append(f'README shows dependency/install info: "{line}"')
#         return "Package registry", "Maven/Gradle ecosystem", evidence

#     # Containers
#     line = _first_matching_line(readme, r"(^|\s)docker\s+(pull|run)\s+")
#     if line:
#         evidence.append(f'README container command: "{line}"')
#         details = "GHCR" if "ghcr.io" in line.lower() else ("Docker Hub" if "docker.io" in line.lower() else None)
#         return "Containers", details, evidence

#     # Builds and releases
#     line = _first_matching_line(readme, r"GitHub\s+Releases|releases\s+page|download\s+the\s+latest\s+release")
#     if line:
#         evidence.append(f'README mentions releases: "{line}"')
#         return "Builds and releases", "GitHub Releases", evidence

#     # Network service
#     line = _first_matching_line(readme, r"(https?://\S+).*(api|endpoint)|REST\s+API|GraphQL|hosted\s+service|web\s+app")
#     if line:
#         evidence.append(f'README indicates network usage: "{line}"')
#         return "Network service", None, evidence

#     # Installer/binary
#     line = _first_matching_line(readme, r"\b(download|installer|binary|exe|dmg|msi)\b")
#     if line:
#         evidence.append(f'README mentions binaries/installers: "{line}"')
#         return "Installer or binary", None, evidence

#     return None, None, evidence


# # ----------------------------
# # Post-check overrides
# # ----------------------------
# _STRONG_GOVERNANCE_HINTS = re.compile(
#     r"(\bgovernance\b|steering committee|technical steering committee|\btsc\b|"
#     r"\bfoundation\b|linux foundation|apache software foundation|cncf|eclipse foundation|"
#     r"\bcharter\b|working group|rfc process|request for comments|"
#     r"maintainers\.md|governance\.md|/governance|/maintainers|"
#     r"code of conduct|contributor covenant)",
#     re.IGNORECASE,
# )

# _INSTITUTIONAL_HINTS = re.compile(
#     r"(\buniversity\b|\bcollege\b|\binstitute\b|\.edu\b|"
#     r"\bnational lab\b|\bresearch lab\b|\bresearch center\b|"
#     r"\bnsf\b|\bnih\b|\bdoe\b|\bnasa\b|grant|funded by|"
#     r"\bacademic\b|\bdepartment of\b)",
#     re.IGNORECASE,
# )

# _COMMERCIAL_HINTS = re.compile(
#     r"(\bcorp\b|\bcorporation\b|\bcompany\b|\bltd\b|\binc\b|\bllc\b|"
#     r"\benterprise\b|\bcommercial\b|\bproduct\b|\bsolutions\b|"
#     r"\bcopyright.*(?:corp|inc|ltd|llc)\b)",
#     re.IGNORECASE,
# )

# _RESEARCH_GROUP_HINTS = re.compile(
#     r"(\blab\b|\bgroup\b|\bresearch group\b|\bresearch team\b|"
#     r"\bPI\b|\bprincipal investigator\b|\bpostdoc\b|\bphd student\b|"
#     r"\bgraduate student\b|\bundergraduate\b)",
#     re.IGNORECASE,
# )

# _LIBRARY_CUES = re.compile(
#     r"(\blibrary\b|\bframework\b|\btoolkit\b|\bpackage\b|\bmodule\b|\bapi\b|"
#     r"\bsdk\b|\bplugin\b|\bextension\b)",
#     re.IGNORECASE,
# )

# _DIRECT_OUTPUT_CUES = re.compile(
#     r"(\bresults\b|\bfigures?\b|\bplots?\b|\bexperiment\b|\bobservations?\b|"
#     r"\bsimulation\b|\banalysis\b|\bvisuali[zs]ation\b|\bdata processing\b|\bpipeline\b|"
#     r"\bspectrum\b|\btime series\b|\btelescope\b|\bmodel\b|\btrain\b|\binference\b)",
#     re.IGNORECASE,
# )

# _APPLICATION_CUES = re.compile(
#     r"(\bcommand.?line\b|\bcli\b|\bgui\b|\bprogram\b|\bapplication\b|"
#     r"\brun\b|\bexecute\b|\blaunch\b)",
#     re.IGNORECASE,
# )


# def _get_owner_type(meta: dict) -> Optional[str]:
#     data = meta.get("data", {})
#     if not isinstance(data, dict):
#         return None
#     owner = data.get("owner", {})
#     if not isinstance(owner, dict):
#         return None
#     t = owner.get("type")
#     return t if isinstance(t, str) else None


# def actor_unit_override(actor_unit: str, meta: dict, readme: str) -> str:
#     """
#     Actor unit override using metadata + README heuristics.
#     """
#     owner_type = _get_owner_type(meta)
#     readme_text = readme or ""

#     # User-owned repos should NOT be Community/Foundation without strong governance
#     if owner_type == "User" and actor_unit == "Community or foundation (open source governance)":
#         if not _STRONG_GOVERNANCE_HINTS.search(readme_text):
#             return "Individual maintainer"

#     # Detect institutional affiliation
#     if actor_unit == "Individual maintainer" and _INSTITUTIONAL_HINTS.search(readme_text):
#         if _RESEARCH_GROUP_HINTS.search(readme_text):
#             return "Research group or lab"
#         return "Institution (university, lab, government research organization)"

#     # Detect commercial entity
#     if actor_unit == "Individual maintainer" and _COMMERCIAL_HINTS.search(readme_text):
#         return "Vendor or commercial entity"

#     # If Unknown and org-owned, try to infer
#     if owner_type == "Organization" and actor_unit == "Unknown":
#         if _RESEARCH_GROUP_HINTS.search(readme_text):
#             return "Research group or lab"
#         if _INSTITUTIONAL_HINTS.search(readme_text):
#             return "Institution (university, lab, government research organization)"
#         if _COMMERCIAL_HINTS.search(readme_text):
#             return "Vendor or commercial entity"

#     return actor_unit


# def research_role_override(research_role: str, supply_chain_role: str, readme: str) -> str:
#     """
#     Research role override based on supply chain role + README cues.
#     """
#     readme_text = readme or ""

#     # Infrastructure/Runtime/Governance are almost always support
#     if supply_chain_role in [
#         "Infrastructure",
#         "Runtime instrumentation software",
#         "Governance software",
#     ]:
#         if research_role == "Direct research execution":
#             return "Research-support tooling"

#     # Dependencies tend to be support unless clearly producing scientific outputs
#     if supply_chain_role == "Dependency software artifact":
#         has_lib = bool(_LIBRARY_CUES.search(readme_text))
#         has_direct = bool(_DIRECT_OUTPUT_CUES.search(readme_text))
#         if has_lib and not has_direct:
#             return "Research-support tooling"

#         # If strongly described as library/toolkit, lean support
#         app_matches = len(_APPLICATION_CUES.findall(readme_text))
#         lib_matches = len(_LIBRARY_CUES.findall(readme_text))
#         if lib_matches > app_matches * 2:
#             return "Research-support tooling"

#     # Applications with direct output cues => direct execution
#     if supply_chain_role == "Application software":
#         if _DIRECT_OUTPUT_CUES.search(readme_text):
#             return "Direct research execution"

#     # If claimed direct but no direct cues, prefer support
#     if research_role == "Direct research execution" and not _DIRECT_OUTPUT_CUES.search(readme_text):
#         if _LIBRARY_CUES.search(readme_text):
#             return "Research-support tooling"

#     return research_role


# # ----------------------------
# # Prompting / GPT call
# # ----------------------------
# def build_prompt(readme: str, repo_metadata: dict) -> str:
#     owner_type = repo_metadata.get("owner_type")
#     owner_login = repo_metadata.get("owner_login")

#     return f"""
# You are labeling a research software artifact using a formal taxonomy.

# You must return exactly one label for each of the four classes:
# 1) actor_unit
# 2) supply_chain_role
# 3) research_role
# 4) distribution_pathway

# Use ONLY README for classes 2, 3, and 4.
# For actor_unit (class 1), you may use README + the provided metadata.

# Provided repo metadata:
# - owner_type: {owner_type}
# - owner_login: {owner_login}

# ===========================
# CLASS 1: Actor unit (who the relevant actor is)
# What it captures:
# The primary organizational unit responsible for producing, maintaining, or operating the research software or its surrounding infrastructure. This dimension helps connect security recommendations to who can realistically implement controls.

# Values and definitions
# • Individual maintainer
# Definition: A single person (or an informal, very small group without formal governance) is the primary developer or maintainer.
# Decision rule: Use when the artifact appears personally maintained, with no clear institutional or community structure.

# • Research group or lab
# Definition: A research lab or project team is the primary producer/maintainer, often centered on a PI, students, or staff within a lab.
# Decision rule: Use when maintenance is tied to a specific research group, grant project, or lab identity.

# • Institution (university, lab, government research organization)
# Definition: A formal institution is responsible for development or operation (for example, university-supported software, national lab software, or institute-managed infrastructure).
# Decision rule: Use when the paper or metadata indicates institutional ownership, institutional hosting, or formal institutional operational responsibility.

# • Community or foundation (open source governance)
# Definition: A broader open source community or foundation provides governance or stewardship (for example, a named community project, foundation-backed ecosystem, or standards body).
# Decision rule: Use when the artifact is maintained under community governance beyond a single lab or institution, especially with formal processes and roles.

# • Vendor or commercial entity
# Definition: A company is the primary producer or operator, including commercial services and proprietary or dual-licensed tools.
# Decision rule: Use when the artifact is primarily developed or operated by a commercial entity.

# • Platform operator (registry or hosting operator)
# Definition: An entity whose primary role is operating distribution or hosting infrastructure used by many projects (for example, package registries, repository hosting, CI platform operators).
# Decision rule: Use when the security-relevant actor is the operator of the distribution or hosting platform rather than a software project maintainer.

# • Mixed or shared responsibility
# Definition: Responsibility is clearly shared across multiple actor units (for example, a lab produces the software but a foundation governs releases, or a community maintains while a vendor operates infrastructure).
# Decision rule: Use when multiple actor units are explicitly implicated.

# • Unknown
# Definition: The paper does not provide enough information to identify the responsible actor unit.

# Notes on how to use it
# This is not “who uses it.” It is “who can change it or enforce controls.”
# If your unit of analysis is a registry policy (like CRAN checks), the actor unit is often Platform operator even if the software packages are maintained by individuals or labs.

# ===========================
# CLASS 2: Supply chain role (README only)
# What it captures:
# Where the artifact sits in the research software supply chain and how a compromise would propagate.

# • Application software
# Definition: A research-facing application that users run to perform a research task (for example, a simulator, analysis program, scientific workflow application).
# Decision rule: If it is primarily executed as an end-user program rather than imported as a dependency, label as Application.

# • Dependency software artifact
# Definition: A reusable library, package, or module intended to be consumed by other software (including transitive dependency components).
# Decision rule: If it is imported, linked, or depended upon by other projects as a component, label as Dependency.

# • Infrastructure
# Definition: Foundational systems that support execution or deployment environments (for example, containerization frameworks, orchestration infrastructure, runtime platforms).
# Decision rule: If it provides an environment or substrate upon which research software runs or is deployed, label as Infrastructure.

# • Runtime instrumentation software
# Definition: Tools that observe, instrument, profile, or modify behavior at runtime for testing, performance, monitoring, or compliance.
# Decision rule: If its primary role is runtime observation or modification rather than producing artifacts or distributing them, label as Runtime instrumentation.

# • Governance software
# Definition: Systems and processes that control or gate distribution to users (for example, package registries, registry policy checks, repository acceptance checks, curated catalogs).
# Decision rule: If it mediates how artifacts enter an ecosystem or are distributed at scale, label as Governance software.

# • Assistant layer software
# Definition: Interactive assistance tooling embedded in developer or community workflows (for example, chat-based assistants integrated into collaboration platforms that guide usage or development).
# Decision rule: If the artifact provides interactive guidance, automation, or decision support within the ecosystem rather than being part of build or distribution, label as Assistant layer.

# • Unknown
# Definition: Insufficient information to determine role.

# ===========================
# CLASS 3: Research coupling or Research role (README only)
# What it captures:
# How directly the software contributes to producing research results versus supporting the research process.

# • Direct research execution
# Definition: Software that directly generates, transforms, or analyzes scientific data or models in a way that contributes to research findings (for example, simulation codes, analysis pipelines, domain tools that produce figures or results).
# Decision rule: If removing the software would prevent producing the core scientific output (results, models, figures), label as Direct.

# • Research-support tooling
# Definition: Software that supports producing research results but is not itself the core analysis or modeling artifact (for example, workflow orchestration, packaging, testing, data management tooling, containerization frameworks, build and release tooling).
# Decision rule: If the software primarily enables development, execution, reproducibility, deployment, or management of research workflows, label as Support.

# • Incidental or general-purpose
# Definition: General software used in research settings but not research-oriented in design or intent (for example, generic OS components, generic editors, general infrastructure unrelated to research goals).
# Decision rule: If the software is broadly used across domains without being research-specific and is not part of a research toolchain intentionally, label as Incidental.

# • Unknown
# Definition: Insufficient information to determine coupling.

# ===========================
# CLASS 4: Distribution pathway (README only)
# What it captures:
# How the software is delivered to downstream users and systems.

# • Source repo
# Definition: Primarily distributed by source code repository, with users building or installing directly from source.
# Decision rule: If the paper emphasizes cloning, building, or installing from a repository without a formal release channel, use Source repo.

# • Builds and releases
# Definition: The software is delivered to downstream users primarily through published build artifacts and releases, such as versioned tags and downloadable assets produced via build and release processes (for example, CI workflows, build scripts, packaging, signing, and publishing). 
# Decision rule: Use Builds and releases when the main unit consumed by users is a published build artifact or release asset, such as a tagged version with attached binaries or archives.

# • Package registry
# Definition: Distributed through a registry or ecosystem package manager (for example, CRAN, PyPI, Maven, npm).
# Decision rule: If installation is described as a single command via an ecosystem registry, use Package registry and specify which.

# • Containers
# Definition: Distributed as container images (for example, Docker or OCI images) or heavily dependent on container-based delivery.
# Decision rule: If the primary operational unit is a container image, use Containers.

# • Installer or binary
# Definition: Distributed as compiled binaries, installers, or downloadable executables.
# Decision rule: If the paper explicitly distinguishes install from binary versus from source, use Installer or binary.

# • Network service
# Definition: Delivered as a hosted or networked service (for example, web apps, APIs, managed platforms).
# Decision rule: If users interact with it over the network rather than installing it, use Network service.

# • Unknown
# Definition: Insufficient information to determine pathway.

# ===========================
# Output requirements:
# - Output must match the schema exactly.
# - Provide evidence bullets grounded in README text/metadata given to you.
# - If uncertain, choose Unknown
# - distribution_details should be null unless pathway is Package registry or Containers.
# ===========================
# README:
# {readme[:120000]}
# """.strip()

# def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
#     resp = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a careful research assistant. Follow the taxonomy rules exactly. Do not invent evidence.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         response_format=SSCResult,
#         temperature=temperature,
#     )

#     msg = resp.choices[0].message
#     if hasattr(msg, "parsed") and msg.parsed is not None:
#         return msg.parsed
#     return SSCResult.model_validate_json(msg.content)


# # ----------------------------
# # Annotation pipeline
# # ----------------------------
# def annotate_one(
#     metadata_path: Path,
#     client: OpenAI,
#     model: str,
#     delay: float,
#     overwrite: bool,
#     dry_run: bool,
# ) -> tuple[bool, str]:
#     try:
#         meta = json.loads(metadata_path.read_text(encoding="utf-8"))
#     except Exception as e:
#         return False, f"bad_json: {e}"

#     if (not overwrite) and ("New_SSC_Taxonomy" in meta):
#         return False, "skip_already_annotated"

#     repo_url = extract_repo_url(meta)
#     if not repo_url:
#         return False, "skip_no_url"

#     readme = fetch_readme(repo_url)
#     if not readme:
#         return False, "skip_no_readme"

#     repo_ctx = extract_repo_context_for_prompt(meta)
#     prompt = build_prompt(readme, repo_ctx)

#     try:
#         result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
#     except Exception as e:
#         return False, f"gpt_error: {e}"

#     # Post-check overrides
#     actor_unit_original = result.actor_unit
#     research_role_original = result.research_role

#     actor_unit = actor_unit_override(result.actor_unit, meta, readme)
#     research_role = research_role_override(result.research_role, result.supply_chain_role, readme)

#     # Deterministic distribution override
#     override_path, override_details, override_evidence = detect_distribution_from_readme(readme)

#     distribution_pathway = result.distribution_pathway
#     distribution_details = result.distribution_details
#     evidence = list(result.evidence or [])

#     if override_path is not None:
#         distribution_pathway = override_path
#         distribution_details = override_details
#         evidence = (override_evidence + evidence)[:3]
#     else:
#         evidence = evidence[:3]

#     overrides_applied = {
#         "actor_unit": actor_unit != actor_unit_original,
#         "research_role": research_role != research_role_original,
#         "distribution_pathway": override_path is not None,
#     }

#     meta["New_SSC_Taxonomy"] = {
#         "actor_unit": actor_unit,
#         "supply_chain_role": result.supply_chain_role,
#         "research_role": research_role,
#         "distribution_pathway": distribution_pathway,
#         "distribution_details": distribution_details,
#         "evidence": evidence,
#         "model": model,
#         "overrides_applied": overrides_applied,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

#     if not dry_run:
#         metadata_path.write_text(
#             json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
#             encoding="utf-8",
#         )

#     time.sleep(delay)
#     return True, "updated" if not dry_run else "dry_run_updated"


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
#     )
#     ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
#     ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo.")
#     ap.add_argument("--model", default="gpt-5.1", help="Model name.")
#     ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit).")
#     ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing classifications.")
#     ap.add_argument("--checkpoint-every", type=int, default=50, help="Log every N files.")
#     ap.add_argument("--dry-run", action="store_true", help="Do not write to disk.")
#     args = ap.parse_args()

#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set.")

#     client = OpenAI(api_key=api_key)

#     repo_root = Path(args.repo_root).expanduser().resolve()
#     db_root = repo_root / "database"

#     providers: List[str]
#     if args.provider == "both":
#         providers = ["github", "gitlab"]
#     else:
#         providers = [args.provider]

#     files: List[Path] = []
#     for p in providers:
#         root = db_root / p
#         if root.exists():
#             files.extend(sorted(root.rglob("metadata.json")))

#     if args.limit and args.limit > 0:
#         files = files[: args.limit]

#     print(f"Repo: {repo_root}")
#     print(f"Provider(s): {providers}")
#     print(f"Model: {args.model}")
#     print(f"Files to process: {len(files)}")
#     if args.dry_run:
#         print("NOTE: dry-run enabled (no files will be modified).")

#     updated = 0
#     skipped = 0
#     errors = 0

#     for i, f in enumerate(tqdm(files, desc="Annotating", unit="repo")):
#         did, status = annotate_one(
#             f,
#             client=client,
#             model=args.model,
#             delay=args.delay,
#             overwrite=args.overwrite,
#             dry_run=args.dry_run,
#         )

#         if did:
#             updated += 1
#         else:
#             if status.startswith("gpt_error") or status.startswith("bad_json"):
#                 errors += 1
#                 print(f"[error] {f}: {status}")
#             else:
#                 skipped += 1

#         if (i + 1) % args.checkpoint_every == 0:
#             print(f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}")

#     print("Done.")
#     print(f"updated={updated} skipped={skipped} errors={errors}")


# if __name__ == "__main__":
#     main()

