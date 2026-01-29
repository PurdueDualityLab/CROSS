#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Literal, Tuple

import requests
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

# ----------------------------
# Allowed label values
# ----------------------------
ACTOR_UNIT_VALUES = [
    "Individual maintainer",
    "Research group or lab",
    "Institution (university, lab, government research organization)",
    "Community or foundation (open source governance)",
    "Vendor or commercial entity",
    "Platform operator (registry or hosting operator)",
    "Mixed or shared responsibility",
    "Unknown",
]

SUPPLY_CHAIN_ROLE_VALUES = [
    "Application software",
    "Dependency software artifact",
    "Infrastructure",
    "Runtime instrumentation software",
    "Governance software",
    "Assistant layer software",
    "Unknown",
]

RESEARCH_ROLE_VALUES = [
    "Direct research execution",
    "Research-support tooling",
    "Incidental or general-purpose",
    "Unknown",
]

DISTRIBUTION_PATHWAY_VALUES = [
    "Source repo",
    "Builds and releases",
    "Package registry",
    "Containers",
    "Installer or binary",
    "Network service",
    "Unknown",
]

# ----------------------------
# Structured output schema
# ----------------------------
class SSCResult(BaseModel):
    actor_unit: Literal[
        "Individual maintainer",
        "Research group or lab",
        "Institution (university, lab, government research organization)",
        "Community or foundation (open source governance)",
        "Vendor or commercial entity",
        "Platform operator (registry or hosting operator)",
        "Mixed or shared responsibility",
        "Unknown",
    ]

    supply_chain_role: Literal[
        "Application software",
        "Dependency software artifact",
        "Infrastructure",
        "Runtime instrumentation software",
        "Governance software",
        "Assistant layer software",
        "Unknown",
    ]

    research_role: Literal[
        "Direct research execution",
        "Research-support tooling",
        "Incidental or general-purpose",
        "Unknown",
    ]

    distribution_pathway: Literal[
        "Source repo",
        "Builds and releases",
        "Package registry",
        "Containers",
        "Installer or binary",
        "Network service",
        "Unknown",
    ]

    distribution_details: Optional[str] = Field(
        default=None,
        description="Optional detail (e.g., PyPI, conda-forge, CRAN, Maven, npm, Docker Hub, GitHub Releases, etc).",
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Up to 3 short quotes/paraphrases from README supporting labels.",
    )


# ----------------------------
# Metadata helpers
# ----------------------------
def extract_repo_url(metadata: dict) -> Optional[str]:
    """
    Schemas vary. Prefer human-facing GitHub URL (not api.github.com).
    """
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

        v = data.get("url")
        if isinstance(v, str) and v.strip() and "github.com" in v and "api.github.com" not in v:
            return v.strip()

    return None


def extract_repo_context_for_prompt(metadata: dict) -> dict:
    """
    Minimal repo metadata that improves actor_unit classification.
    """
    data = metadata.get("data", {})
    owner = {}
    if isinstance(data, dict):
        owner = data.get("owner") or {}

    def _safe_str(x):
        return x if isinstance(x, str) else None

    ctx = {
        "uid": _safe_str(metadata.get("uid")),
        "parser": _safe_str(metadata.get("parser")),
        "html_url": _safe_str(data.get("html_url")) if isinstance(data, dict) else None,
        "full_name": _safe_str(data.get("full_name")) if isinstance(data, dict) else None,
        "name": _safe_str(data.get("name")) if isinstance(data, dict) else None,
        "description": _safe_str(data.get("description")) if isinstance(data, dict) else None,
        "homepage": _safe_str(data.get("homepage")) if isinstance(data, dict) else None,
        "topics": data.get("topics") if isinstance(data, dict) else None,
        "owner_login": _safe_str(owner.get("login")) if isinstance(owner, dict) else None,
        "owner_type": _safe_str(owner.get("type")) if isinstance(owner, dict) else None,
        "owner_html_url": _safe_str(owner.get("html_url")) if isinstance(owner, dict) else None,
    }

    return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}


# ----------------------------
# README fetching (no GitHub token)
# ----------------------------
def github_raw_readme_urls(repo_url: str) -> List[str]:
    """
    Try common README locations via raw.githubusercontent.com.
    """
    repo_url = repo_url.rstrip("/")
    parts = repo_url.split("/")
    if len(parts) < 2:
        return []

    owner, repo = parts[-2], parts[-1]
    base = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/"
    candidates = [
        "README.md",
        "README.MD",
        "README",
        "readme.md",
        "Readme.md",
        "docs/README.md",
        "README.rst",
        "README.txt",
        "DOCS.md",
        "docs/index.md",
    ]
    return [base + c for c in candidates]


def fetch_readme(repo_url: str, timeout: int = 15) -> Optional[str]:
    if not repo_url or "github.com" not in repo_url:
        return None

    headers = {"User-Agent": "rse-ssc-annotator/1.0"}

    for raw in github_raw_readme_urls(repo_url):
        try:
            r = requests.get(raw, timeout=timeout, headers=headers)
            if r.status_code == 200 and r.text and r.text.strip():
                return r.text
        except Exception:
            continue

    return None


# ----------------------------
# Distribution pathway deterministic detector
# ----------------------------
def _first_matching_line(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
    rx = re.compile(pattern, flags)
    for line in text.splitlines():
        if rx.search(line):
            line = line.strip()
            if line:
                return line[:200]
    return None


def detect_distribution_from_readme(readme: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Returns (pathway, details, evidence_snippets).
    Evidence snippets are auditable from README.
    """
    evidence: List[str] = []
    if not readme:
        return None, None, evidence

    # Package registries
    line = _first_matching_line(readme, r"(^|\s)pip\s+install\s+")
    if line:
        evidence.append(f'README install command: "{line}"')
        return "Package registry", "PyPI", evidence

    line = _first_matching_line(readme, r"(^|\s)conda\s+install\s+")
    if line:
        details = "conda-forge" if "conda-forge" in line.lower() else "conda"
        evidence.append(f'README install command: "{line}"')
        return "Package registry", details, evidence

    line = _first_matching_line(readme, r"install\.packages\(|remotes::install_")
    if line:
        evidence.append(f'README install command: "{line}"')
        return "Package registry", "CRAN (or R ecosystem)", evidence

    line = _first_matching_line(readme, r"Pkg\.add\(")
    if line:
        evidence.append(f'README install command: "{line}"')
        return "Package registry", "Julia General registry", evidence

    line = _first_matching_line(readme, r"(^|\s)(npm\s+install|yarn\s+add)\s+")
    if line:
        evidence.append(f'README install command: "{line}"')
        return "Package registry", "npm", evidence

    line = _first_matching_line(readme, r'(<dependency>|implementation\s+["\']|mvn\s+)')
    if line:
        evidence.append(f'README shows dependency/install info: "{line}"')
        return "Package registry", "Maven/Gradle ecosystem", evidence

    # Containers
    line = _first_matching_line(readme, r"(^|\s)docker\s+(pull|run)\s+")
    if line:
        evidence.append(f'README container command: "{line}"')
        details = "GHCR" if "ghcr.io" in line.lower() else ("Docker Hub" if "docker.io" in line.lower() else None)
        return "Containers", details, evidence

    # Builds and releases
    line = _first_matching_line(readme, r"GitHub\s+Releases|releases\s+page|download\s+the\s+latest\s+release")
    if line:
        evidence.append(f'README mentions releases: "{line}"')
        return "Builds and releases", "GitHub Releases", evidence

    # Network service
    line = _first_matching_line(readme, r"(https?://\S+).*(api|endpoint)|REST\s+API|GraphQL|hosted\s+service|web\s+app")
    if line:
        evidence.append(f'README indicates network usage: "{line}"')
        return "Network service", None, evidence

    # Installer/binary
    line = _first_matching_line(readme, r"\b(download|installer|binary|exe|dmg|msi)\b")
    if line:
        evidence.append(f'README mentions binaries/installers: "{line}"')
        return "Installer or binary", None, evidence

    return None, None, evidence


# ----------------------------
# Post-check overrides
# ----------------------------
_STRONG_GOVERNANCE_HINTS = re.compile(
    r"(\bgovernance\b|steering committee|technical steering committee|\btsc\b|"
    r"\bfoundation\b|linux foundation|apache software foundation|cncf|eclipse foundation|"
    r"\bcharter\b|working group|rfc process|request for comments|"
    r"maintainers\.md|governance\.md|/governance|/maintainers|"
    r"code of conduct|contributor covenant)",
    re.IGNORECASE,
)

_INSTITUTIONAL_HINTS = re.compile(
    r"(\buniversity\b|\bcollege\b|\binstitute\b|\.edu\b|"
    r"\bnational lab\b|\bresearch lab\b|\bresearch center\b|"
    r"\bnsf\b|\bnih\b|\bdoe\b|\bnasa\b|grant|funded by|"
    r"\bacademic\b|\bdepartment of\b)",
    re.IGNORECASE,
)

_COMMERCIAL_HINTS = re.compile(
    r"(\bcorp\b|\bcorporation\b|\bcompany\b|\bltd\b|\binc\b|\bllc\b|"
    r"\benterprise\b|\bcommercial\b|\bproduct\b|\bsolutions\b|"
    r"\bcopyright.*(?:corp|inc|ltd|llc)\b)",
    re.IGNORECASE,
)

_RESEARCH_GROUP_HINTS = re.compile(
    r"(\blab\b|\bgroup\b|\bresearch group\b|\bresearch team\b|"
    r"\bPI\b|\bprincipal investigator\b|\bpostdoc\b|\bphd student\b|"
    r"\bgraduate student\b|\bundergraduate\b)",
    re.IGNORECASE,
)

_LIBRARY_CUES = re.compile(
    r"(\blibrary\b|\bframework\b|\btoolkit\b|\bpackage\b|\bmodule\b|\bapi\b|"
    r"\bsdk\b|\bplugin\b|\bextension\b)",
    re.IGNORECASE,
)

_DIRECT_OUTPUT_CUES = re.compile(
    r"(\bresults\b|\bfigures?\b|\bplots?\b|\bexperiment\b|\bobservations?\b|"
    r"\bsimulation\b|\banalysis\b|\bvisuali[zs]ation\b|\bdata processing\b|\bpipeline\b|"
    r"\bspectrum\b|\btime series\b|\btelescope\b|\bmodel\b|\btrain\b|\binference\b)",
    re.IGNORECASE,
)

_APPLICATION_CUES = re.compile(
    r"(\bcommand.?line\b|\bcli\b|\bgui\b|\bprogram\b|\bapplication\b|"
    r"\brun\b|\bexecute\b|\blaunch\b)",
    re.IGNORECASE,
)


def _get_owner_type(meta: dict) -> Optional[str]:
    data = meta.get("data", {})
    if not isinstance(data, dict):
        return None
    owner = data.get("owner", {})
    if not isinstance(owner, dict):
        return None
    t = owner.get("type")
    return t if isinstance(t, str) else None


def actor_unit_override(actor_unit: str, meta: dict, readme: str) -> str:
    """
    Actor unit override using metadata + README heuristics.
    """
    owner_type = _get_owner_type(meta)
    readme_text = readme or ""

    # User-owned repos should NOT be Community/Foundation without strong governance
    if owner_type == "User" and actor_unit == "Community or foundation (open source governance)":
        if not _STRONG_GOVERNANCE_HINTS.search(readme_text):
            return "Individual maintainer"

    # Detect institutional affiliation
    if actor_unit == "Individual maintainer" and _INSTITUTIONAL_HINTS.search(readme_text):
        if _RESEARCH_GROUP_HINTS.search(readme_text):
            return "Research group or lab"
        return "Institution (university, lab, government research organization)"

    # Detect commercial entity
    if actor_unit == "Individual maintainer" and _COMMERCIAL_HINTS.search(readme_text):
        return "Vendor or commercial entity"

    # If Unknown and org-owned, try to infer
    if owner_type == "Organization" and actor_unit == "Unknown":
        if _RESEARCH_GROUP_HINTS.search(readme_text):
            return "Research group or lab"
        if _INSTITUTIONAL_HINTS.search(readme_text):
            return "Institution (university, lab, government research organization)"
        if _COMMERCIAL_HINTS.search(readme_text):
            return "Vendor or commercial entity"

    return actor_unit


def research_role_override(research_role: str, supply_chain_role: str, readme: str) -> str:
    """
    Research role override based on supply chain role + README cues.
    """
    readme_text = readme or ""

    # Infrastructure/Runtime/Governance are almost always support
    if supply_chain_role in [
        "Infrastructure",
        "Runtime instrumentation software",
        "Governance software",
    ]:
        if research_role == "Direct research execution":
            return "Research-support tooling"

    # Dependencies tend to be support unless clearly producing scientific outputs
    if supply_chain_role == "Dependency software artifact":
        has_lib = bool(_LIBRARY_CUES.search(readme_text))
        has_direct = bool(_DIRECT_OUTPUT_CUES.search(readme_text))
        if has_lib and not has_direct:
            return "Research-support tooling"

        # If strongly described as library/toolkit, lean support
        app_matches = len(_APPLICATION_CUES.findall(readme_text))
        lib_matches = len(_LIBRARY_CUES.findall(readme_text))
        if lib_matches > app_matches * 2:
            return "Research-support tooling"

    # Applications with direct output cues => direct execution
    if supply_chain_role == "Application software":
        if _DIRECT_OUTPUT_CUES.search(readme_text):
            return "Direct research execution"

    # If claimed direct but no direct cues, prefer support
    if research_role == "Direct research execution" and not _DIRECT_OUTPUT_CUES.search(readme_text):
        if _LIBRARY_CUES.search(readme_text):
            return "Research-support tooling"

    return research_role


# ----------------------------
# Prompting / GPT call
# ----------------------------
def build_prompt(readme: str, repo_metadata: dict) -> str:
    owner_type = repo_metadata.get("owner_type")
    owner_login = repo_metadata.get("owner_login")

    return f"""
You are labeling a research software artifact using a formal taxonomy.

You must return exactly one label for each of the four classes:
1) actor_unit
2) supply_chain_role
3) research_role
4) distribution_pathway

Use ONLY README for classes 2, 3, and 4.
For actor_unit (class 1), you may use README + the provided metadata.

Provided repo metadata:
- owner_type: {owner_type}
- owner_login: {owner_login}

===========================
CLASS 1: Actor unit (who the relevant actor is)
What it captures:
The primary organizational unit responsible for producing, maintaining, or operating the research software or its surrounding infrastructure. This dimension helps connect security recommendations to who can realistically implement controls.

Values and definitions
• Individual maintainer
Definition: A single person (or an informal, very small group without formal governance) is the primary developer or maintainer.
Decision rule: Use when the artifact appears personally maintained, with no clear institutional or community structure.

• Research group or lab
Definition: A research lab or project team is the primary producer/maintainer, often centered on a PI, students, or staff within a lab.
Decision rule: Use when maintenance is tied to a specific research group, grant project, or lab identity.

• Institution (university, lab, government research organization)
Definition: A formal institution is responsible for development or operation (for example, university-supported software, national lab software, or institute-managed infrastructure).
Decision rule: Use when the paper or metadata indicates institutional ownership, institutional hosting, or formal institutional operational responsibility.

• Community or foundation (open source governance)
Definition: A broader open source community or foundation provides governance or stewardship (for example, a named community project, foundation-backed ecosystem, or standards body).
Decision rule: Use when the artifact is maintained under community governance beyond a single lab or institution, especially with formal processes and roles.

• Vendor or commercial entity
Definition: A company is the primary producer or operator, including commercial services and proprietary or dual-licensed tools.
Decision rule: Use when the artifact is primarily developed or operated by a commercial entity.

• Platform operator (registry or hosting operator)
Definition: An entity whose primary role is operating distribution or hosting infrastructure used by many projects (for example, package registries, repository hosting, CI platform operators).
Decision rule: Use when the security-relevant actor is the operator of the distribution or hosting platform rather than a software project maintainer.

• Mixed or shared responsibility
Definition: Responsibility is clearly shared across multiple actor units (for example, a lab produces the software but a foundation governs releases, or a community maintains while a vendor operates infrastructure).
Decision rule: Use when multiple actor units are explicitly implicated.

• Unknown
Definition: The paper does not provide enough information to identify the responsible actor unit.

Notes on how to use it
This is not “who uses it.” It is “who can change it or enforce controls.”
If your unit of analysis is a registry policy (like CRAN checks), the actor unit is often Platform operator even if the software packages are maintained by individuals or labs.

===========================
CLASS 2: Supply chain role (README only)
What it captures:
Where the artifact sits in the research software supply chain and how a compromise would propagate.

• Application software
Definition: A research-facing application that users run to perform a research task (for example, a simulator, analysis program, scientific workflow application).
Decision rule: If it is primarily executed as an end-user program rather than imported as a dependency, label as Application.

• Dependency software artifact
Definition: A reusable library, package, or module intended to be consumed by other software (including transitive dependency components).
Decision rule: If it is imported, linked, or depended upon by other projects as a component, label as Dependency.

• Infrastructure
Definition: Foundational systems that support execution or deployment environments (for example, containerization frameworks, orchestration infrastructure, runtime platforms).
Decision rule: If it provides an environment or substrate upon which research software runs or is deployed, label as Infrastructure.

• Runtime instrumentation software
Definition: Tools that observe, instrument, profile, or modify behavior at runtime for testing, performance, monitoring, or compliance.
Decision rule: If its primary role is runtime observation or modification rather than producing artifacts or distributing them, label as Runtime instrumentation.

• Governance software
Definition: Systems and processes that control or gate distribution to users (for example, package registries, registry policy checks, repository acceptance checks, curated catalogs).
Decision rule: If it mediates how artifacts enter an ecosystem or are distributed at scale, label as Governance software.

• Assistant layer software
Definition: Interactive assistance tooling embedded in developer or community workflows (for example, chat-based assistants integrated into collaboration platforms that guide usage or development).
Decision rule: If the artifact provides interactive guidance, automation, or decision support within the ecosystem rather than being part of build or distribution, label as Assistant layer.

• Unknown
Definition: Insufficient information to determine role.

===========================
CLASS 3: Research coupling or Research role (README only)
What it captures:
How directly the software contributes to producing research results versus supporting the research process.

• Direct research execution
Definition: Software that directly generates, transforms, or analyzes scientific data or models in a way that contributes to research findings (for example, simulation codes, analysis pipelines, domain tools that produce figures or results).
Decision rule: If removing the software would prevent producing the core scientific output (results, models, figures), label as Direct.

• Research-support tooling
Definition: Software that supports producing research results but is not itself the core analysis or modeling artifact (for example, workflow orchestration, packaging, testing, data management tooling, containerization frameworks, build and release tooling).
Decision rule: If the software primarily enables development, execution, reproducibility, deployment, or management of research workflows, label as Support.

• Incidental or general-purpose
Definition: General software used in research settings but not research-oriented in design or intent (for example, generic OS components, generic editors, general infrastructure unrelated to research goals).
Decision rule: If the software is broadly used across domains without being research-specific and is not part of a research toolchain intentionally, label as Incidental.

• Unknown
Definition: Insufficient information to determine coupling.

===========================
CLASS 4: Distribution pathway (README only)
What it captures:
How the software is delivered to downstream users and systems.

• Source repo
Definition: Primarily distributed by source code repository, with users building or installing directly from source.
Decision rule: If the paper emphasizes cloning, building, or installing from a repository without a formal release channel, use Source repo.

• Builds and releases
Definition: The software is delivered to downstream users primarily through published build artifacts and releases, such as versioned tags and downloadable assets produced via build and release processes (for example, CI workflows, build scripts, packaging, signing, and publishing). 
Decision rule: Use Builds and releases when the main unit consumed by users is a published build artifact or release asset, such as a tagged version with attached binaries or archives.

• Package registry
Definition: Distributed through a registry or ecosystem package manager (for example, CRAN, PyPI, Maven, npm).
Decision rule: If installation is described as a single command via an ecosystem registry, use Package registry and specify which.

• Containers
Definition: Distributed as container images (for example, Docker or OCI images) or heavily dependent on container-based delivery.
Decision rule: If the primary operational unit is a container image, use Containers.

• Installer or binary
Definition: Distributed as compiled binaries, installers, or downloadable executables.
Decision rule: If the paper explicitly distinguishes install from binary versus from source, use Installer or binary.

• Network service
Definition: Delivered as a hosted or networked service (for example, web apps, APIs, managed platforms).
Decision rule: If users interact with it over the network rather than installing it, use Network service.

• Unknown
Definition: Insufficient information to determine pathway.

===========================
Output requirements:
- Output must match the schema exactly.
- Provide evidence bullets grounded in README text/metadata given to you.
- If uncertain, choose Unknown
- distribution_details should be null unless pathway is Package registry or Containers.
===========================
README:
{readme[:120000]}
""".strip()

def call_gpt(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> SSCResult:
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful research assistant. Follow the taxonomy rules exactly. Do not invent evidence.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=SSCResult,
        temperature=temperature,
    )

    msg = resp.choices[0].message
    if hasattr(msg, "parsed") and msg.parsed is not None:
        return msg.parsed
    return SSCResult.model_validate_json(msg.content)


# ----------------------------
# Annotation pipeline
# ----------------------------
def annotate_one(
    metadata_path: Path,
    client: OpenAI,
    model: str,
    model_key: str,
    delay: float,
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"bad_json: {e}"

    # Initialize New_SSC_Taxonomy as dict if it doesn't exist
    if "New_SSC_Taxonomy" not in meta:
        meta["New_SSC_Taxonomy"] = {}

    # Check if this specific model already ran
    if (not overwrite) and (model_key in meta["New_SSC_Taxonomy"]):
        return False, f"skip_already_annotated_{model_key}"

    repo_url = extract_repo_url(meta)
    if not repo_url:
        return False, "skip_no_url"

    readme = fetch_readme(repo_url)
    if not readme:
        return False, "skip_no_readme"

    repo_ctx = extract_repo_context_for_prompt(meta)
    prompt = build_prompt(readme, repo_ctx)

    try:
        result = call_gpt(client, model=model, prompt=prompt, temperature=0.0)
    except Exception as e:
        return False, f"gpt_error: {e}"

    # Post-check overrides
    actor_unit_original = result.actor_unit
    research_role_original = result.research_role

    actor_unit = actor_unit_override(result.actor_unit, meta, readme)
    research_role = research_role_override(result.research_role, result.supply_chain_role, readme)

    # Deterministic distribution override
    override_path, override_details, override_evidence = detect_distribution_from_readme(readme)

    distribution_pathway = result.distribution_pathway
    distribution_details = result.distribution_details
    evidence = list(result.evidence or [])

    if override_path is not None:
        distribution_pathway = override_path
        distribution_details = override_details
        evidence = (override_evidence + evidence)[:3]
    else:
        evidence = evidence[:3]

    overrides_applied = {
        "actor_unit": actor_unit != actor_unit_original,
        "research_role": research_role != research_role_original,
        "distribution_pathway": override_path is not None,
    }

    # Store under model-specific key
    meta["New_SSC_Taxonomy"][model_key] = {
        "actor_unit": actor_unit,
        "supply_chain_role": result.supply_chain_role,
        "research_role": research_role,
        "distribution_pathway": distribution_pathway,
        "distribution_details": distribution_details,
        "evidence": evidence,
        "model": model,
        "overrides_applied": overrides_applied,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not dry_run:
        metadata_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    time.sleep(delay)
    return True, f"updated_{model_key}" if not dry_run else f"dry_run_updated_{model_key}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate rseng/software metadata.json files with SSC taxonomy labels."
    )
    ap.add_argument("--provider", choices=["github", "gitlab", "both"], default="github")
    ap.add_argument("--repo-root", default=".", help="Path to rseng/software repo.")
    ap.add_argument("--model", default="gpt-5.1", help="Model name.")
    ap.add_argument("--model-name", default=None, help="Key for storing results (default: use model name)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit).")
    ap.add_argument("--delay", type=float, default=0.25, help="Delay between GPT calls (seconds).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing classifications for this model.")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Log every N files.")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Process files in fixed-size batches (0 = no batching).",
    )
    ap.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Which batch to process (0-based). Used only when --batch-size > 0.",
    )
    ap.add_argument(
        "--auto-batch",
        action="store_true",
        help="Process all batches sequentially when --batch-size > 0.",
    )
    ap.add_argument(
        "--progress-file",
        default=None,
        help="Path to a JSONL file that records completed file paths for resume.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already listed in --progress-file.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write to disk.")
    args = ap.parse_args()

    # Use model name as key if not specified
    model_key = args.model_name if args.model_name else args.model

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

    repo_root = Path(args.repo_root).expanduser().resolve()
    db_root = repo_root / "database"

    providers: List[str]
    if args.provider == "both":
        providers = ["github", "gitlab"]
    else:
        providers = [args.provider]

    files: List[Path] = []
    for p in providers:
        root = db_root / p
        if root.exists():
            files.extend(sorted(root.rglob("metadata.json")))

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if args.auto_batch and not (args.batch_size and args.batch_size > 0):
        raise ValueError("--auto-batch requires --batch-size > 0")

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
                if (
                    isinstance(path, str)
                    and isinstance(status, str)
                    and not status.startswith("gpt_error")
                    and not status.startswith("bad_json")
                    and not status.startswith("error")
                ):
                    completed.add(path)

    print(f"Repo: {repo_root}")
    print(f"Provider(s): {providers}")
    print(f"Model: {args.model}")
    print(f"Model key: {model_key}")
    print(f"Files to process: {len(files)}")
    if args.batch_size and args.batch_size > 0:
        if args.auto_batch:
            print(f"Batch: auto size={args.batch_size}")
        else:
            print(f"Batch: index={args.batch_index} size={args.batch_size}")
    if args.progress_file:
        print(f"Progress file: {args.progress_file}")
    
    if args.dry_run:
        print("NOTE: dry-run enabled (no files will be modified).")

    updated = 0
    skipped = 0
    errors = 0

    def process_files(file_list: List[Path], desc: str) -> None:
        nonlocal updated, skipped, errors
        if args.resume and completed:
            file_list = [f for f in file_list if str(f) not in completed]
        for i, f in enumerate(tqdm(file_list, desc=desc, unit="repo")):
            try:
                did, status = annotate_one(
                    f,
                    client=client,
                    model=args.model,
                    model_key=model_key,  
                    delay=args.delay,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                did = False
                status = f"error: {e}"

            if did:
                updated += 1
            else:
                if status.startswith("gpt_error") or status.startswith("bad_json") or status.startswith("error"):
                    errors += 1
                    print(f"[error] {f}: {status}")
                else:
                    skipped += 1
            if args.progress_file:
                progress_path = Path(args.progress_file).expanduser()
                record = {"path": str(f), "status": status}
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % args.checkpoint_every == 0:
                print(
                    f"[checkpoint] processed={i+1} updated={updated} skipped={skipped} errors={errors}"
                )

    if args.batch_size and args.batch_size > 0:
        if args.auto_batch:
            total_batches = (len(files) + args.batch_size - 1) // args.batch_size
            for batch_index in range(total_batches):
                start = batch_index * args.batch_size
                end = min(start + args.batch_size, len(files))
                batch_files = files[start:end]
                desc = f"Annotating ({model_key}) [batch {batch_index+1}/{total_batches}]"
                process_files(batch_files, desc)
        else:
            if args.batch_index < 0:
                raise ValueError("--batch-index must be >= 0")
            total_batches = (len(files) + args.batch_size - 1) // args.batch_size
            if args.batch_index >= total_batches:
                raise ValueError(
                    f"--batch-index out of range: {args.batch_index} >= {total_batches}"
                )
            start = args.batch_index * args.batch_size
            end = min(start + args.batch_size, len(files))
            files = files[start:end]
            process_files(files, f"Annotating ({model_key})")
    else:
        process_files(files, f"Annotating ({model_key})")

    print("Done.")
    print(f"updated={updated} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
