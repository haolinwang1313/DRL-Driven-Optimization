from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from paper_repro.config import Config
from paper_repro.contracts import write_json


def load_private_env_file(env_file: Path | None = None) -> dict[str, str]:
    target = env_file or (Path.home() / ".claude" / ".env")
    loaded: dict[str, str] = {}
    if not target.exists():
        return loaded
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            loaded[key] = value
    return loaded


def get_claude_env() -> dict[str, str]:
    env = load_private_env_file()
    return {
        "api_key": env.get("CLAUDE_REVIEW_API_KEY", ""),
        "base_url": env.get("CLAUDE_REVIEW_BASE_URL", "https://api.openai.com/v1"),
        "model": env.get("CLAUDE_REVIEW_MODEL", "claude-opus-4-6"),
    }


def call_claude_review(prompt: str, *, model: str, base_url: str, api_key: str, max_tokens: int = 4096, retries: int = 3) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8", errors="replace")
            raw = json.loads(body)
            return raw
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == retries:
                raise RuntimeError(last_error) from exc
            time.sleep(attempt * 2)
    raise RuntimeError(last_error or "unknown Claude API error")


def extract_response_text(raw_payload: dict[str, Any]) -> str:
    choices = raw_payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def build_revision_review_prompt(config: Config) -> str:
    diagnostics_path = Path(config["publication"]["diagnostics_dir"]) / "publication_diagnostics.json"
    reevaluation_path = Path(config["publication"]["reevaluation_dir"]) / "top_candidate_reevaluation.csv"
    tracker_path = Path(config["publication"]["tracker_path"])
    manuscript_path = Path("elsarticle") / "manuscript.tex"

    diagnostics_text = diagnostics_path.read_text(encoding="utf-8") if diagnostics_path.exists() else "{}"
    tracker_text = tracker_path.read_text(encoding="utf-8") if tracker_path.exists() else "{}"
    reevaluation_excerpt = reevaluation_path.read_text(encoding="utf-8")[:4000] if reevaluation_path.exists() else ""
    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    anchor_lines = []
    for needle in [
        "The DRL framework substantially outperformed NSGA-II",
        "NSGA-II achieves the highest Hypervolume and the lowest IGD",
        "The surrogate is trained on only 500 simulated cases",
        "no external empirical validation dataset is included",
        "\\section*{Abbreviations}",
    ]:
        if needle in manuscript_text:
            anchor_lines.append(needle)

    return f"""You are a senior Applied Energy technical reviewer.

Assess whether the CURRENT CODE-SIDE REVISION materially addresses the reviewers' technical concerns for a DRL + surrogate urban morphology paper.

Current diagnostics summary:
{diagnostics_text}

Top-candidate re-evaluation excerpt:
{reevaluation_excerpt}

Revision tracker summary:
{tracker_text[:6000]}

Key manuscript anchors now present:
{chr(10).join('- ' + line for line in anchor_lines)}

Important context:
- The current imported publication artifacts still report simulation_mode=fallback_analytic.
- Current HV/IGD summary favors NSGA-II over all DRL scenarios.
- The manuscript has already been rewritten to remove the universal-DRL-superiority claim.
- The supplement now includes AF/OSR consistency, CV caption, nonlinear response diagnostics, and candidate re-evaluation.

Return only four sections:
A. Closed reviewer points
B. Partially addressed reviewer points
C. Top 5 remaining technical blockers
D. Minimum next evidence needed before resubmission

Focus only on methodology, validation, benchmark fairness, convergence, figure/table completeness, supplement consistency, and claim-evidence alignment. No language editing.
"""


def run_revision_review(config: Config) -> dict[str, Any]:
    env = get_claude_env()
    if not env["api_key"]:
        raise RuntimeError("CLAUDE_REVIEW_API_KEY not found in ~/.claude/.env")
    prompt = build_revision_review_prompt(config)
    raw = call_claude_review(prompt, model=env["model"], base_url=env["base_url"], api_key=env["api_key"])
    response_text = extract_response_text(raw)
    output_dir = Path(config["report"]["reports_dir"]) / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw, output_dir / "claude_revision_review_raw.json")
    (output_dir / "claude_revision_review.md").write_text(response_text, encoding="utf-8")
    return {
        "model": env["model"],
        "base_url": env["base_url"],
        "review_path": str(output_dir / "claude_revision_review.md"),
        "raw_path": str(output_dir / "claude_revision_review_raw.json"),
    }
