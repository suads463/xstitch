"""Relevance-based task discovery engine for Stitch.

Inspired by PageIndex (https://github.com/VectifyAI/PageIndex):
  RELEVANCE ≠ SIMILARITY — matching by reasoning about structure, not distance.

PageIndex uses LLM reasoning over a hierarchical tree index to navigate from
broad (table of contents) to specific (section content). We cannot use LLM calls
(zero deps, no API key), but we apply the same structural principles:

  1. Hierarchical two-phase scoring — coarse fields (title/objective) first,
     deep fields (decisions/snapshots) second, with a multi-level confirmation
     boost when BOTH levels agree (mirrors PageIndex tree traversal).
  2. Stemming + synonym expansion — approximate semantic understanding. Bridges
     vocabulary gaps: "auth" matches "authentication", "db" matches "database".
  3. BM25 Okapi — term rarity (IDF) matters. Rare terms are strong relevance
     signals; common terms are noise. This is fundamentally different from
     cosine similarity where all dimensions contribute equally.
  4. Bigram matching — compound terms ("rate_limit") are treated as strong,
     specific signals with high weight.
  5. Recency decay — recently updated tasks score higher. Time is a relevance
     signal: a task touched 1 hour ago is more likely what the user means.

Zero external dependencies — stdlib math only.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .store import Store
from .models import Task, Snapshot, Decision
from .capture import run_git, is_git_repo


# ---------------------------------------------------------------------------
# Stemming — lightweight suffix stripping for developer vocabulary
# ---------------------------------------------------------------------------

_STEM_RULES = [
    ("ization", 3), ("isation", 3), ("ational", 3),
    ("ation", 3), ("ition", 3), ("ution", 3),
    ("ement", 3), ("ment", 3),
    ("ness", 3), ("ity", 3),
    ("ible", 3), ("able", 3),
    ("ful", 3), ("less", 4), ("ive", 3), ("ous", 3),
    ("ing", 3), ("ion", 3),
    ("ied", 3), ("ies", 3),
    ("ate", 3),
    ("ed", 3), ("er", 3), ("es", 3),
    ("ly", 3), ("al", 3),
    ("y", 4), ("s", 4),
]


def _stem(word: str) -> str:
    """Reduce a word to its approximate root via suffix stripping.

    Not linguistically perfect — trades precision for recall. For task
    matching in a small corpus, recall (finding the right task) matters
    far more than producing correct dictionary roots.
    """
    if len(word) <= 4:
        return word
    for suffix, min_remaining in _STEM_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_remaining:
            return word[:-len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Synonym / Alias Expansion — bridges abbreviation and vocabulary gaps
# ---------------------------------------------------------------------------

ALIAS_GROUPS: list[set[str]] = [
    {"db", "databas", "database"},
    {"auth", "authentication", "authorization", "authenticate"},
    {"api", "endpoint"},
    {"config", "configuration", "configure", "conf"},
    {"k8s", "kubernetes", "kube"},
    {"infra", "infrastructure"},
    {"repo", "repository"},
    {"impl", "implementation", "implement"},
    {"perf", "performance"},
    {"dep", "deps", "dependency"},
    {"env", "environment"},
    {"doc", "docs", "documentation"},
    {"err", "error"},
    {"req", "request"},
    {"res", "response"},
    {"pkg", "package"},
    {"fn", "func", "function"},
    {"param", "params", "parameter"},
    {"postgres", "postgresql", "psql"},
    {"mongo", "mongodb"},
    {"js", "javascript"},
    {"ts", "typescript"},
    {"py", "python"},
    {"msg", "message"},
    {"async", "asynchronous"},
    {"sync", "synchronous"},
    {"dir", "directory"},
    {"val", "validation", "validate"},
    {"init", "initialization", "initialize"},
    {"gen", "generate", "generation", "generator"},
    {"migr", "migrat", "migrate", "migration"},
    {"deploy", "deployment"},
    {"test", "testing"},
    {"debug", "debugg", "debugging"},
    {"refactor", "refactoring"},
    {"cache", "cach", "caching"},
    {"queue", "queuing"},
    {"svc", "service"},
    {"ctrl", "controller"},
    {"mw", "middleware"},
    {"ws", "websocket"},
]

# Build bidirectional alias map with auto-stem expansion.
# For each group, also index the stemmed forms of every term so that
# inflected forms (e.g., "migrating" → stem "migrat") connect back
# to the alias group.
ALIAS_MAP: dict[str, set[str]] = {}
for _group in ALIAS_GROUPS:
    _all_forms: set[str] = set()
    for _term in _group:
        _all_forms.add(_term)
        _s = _stem(_term)
        if _s != _term:
            _all_forms.add(_s)
    for _term in _all_forms:
        ALIAS_MAP.setdefault(_term, set()).update(_all_forms - {_term})


# ---------------------------------------------------------------------------
# Bigram extraction — compound terms are stronger signals
# ---------------------------------------------------------------------------

def _extract_bigrams(tokens: list[str]) -> list[str]:
    """Generate compound bigram tokens from adjacent pairs.

    "rate" + "limit" → "rate_limit". These are highly specific signals:
    matching a bigram is much stronger evidence than matching either unigram.
    """
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]


# ---------------------------------------------------------------------------
# Recency decay — time is a relevance signal
# ---------------------------------------------------------------------------

def _time_decay_factor(updated_at: str) -> float:
    """Compute recency boost with 30-day half-life.

    Tasks updated recently are more likely what the user is referring to.
    Floor at 0.1 so very old tasks can still match on strong term signals.
    """
    try:
        dt = datetime.fromisoformat(updated_at)
        now = datetime.now(timezone.utc)
        days_old = max((now - dt).total_seconds() / 86400, 0)
        return max(0.5 ** (days_old / 30), 0.1)
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# Tokenization — stem, expand aliases, deduplicate
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "i", "me", "my", "we", "our", "the", "a", "an", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "can", "may", "might",
    "shall", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "about", "that", "this", "it", "its", "and", "or",
    "but", "if", "then", "so", "up", "out", "no", "not", "what", "which",
    "who", "when", "where", "how", "all", "each", "every", "both", "few",
    "more", "most", "some", "any", "let", "lets", "want", "need", "please",
    "work", "working", "task", "project", "also", "just", "like", "get",
    "make", "done", "thing", "things", "way", "start", "started", "using",
    "use", "used", "try", "tried", "help", "still", "back", "going",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for relevance matching.

    Pipeline: split → remove stop words → stem → expand aliases.

    Alias-expanded terms get TF=1 per expansion (weaker than explicit
    mentions), which is semantically correct: a document mentioning
    "database" 5 times has TF=5 for "databas" but only TF=1 for the
    alias "db".
    """
    if not text:
        return []

    # Split camelCase before lowercasing (handleAuth → handle auth)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-z0-9]+", text.lower())

    tokens = []
    seen_aliases: set[str] = set()

    for w in words:
        if w in STOP_WORDS or len(w) < 2:
            continue

        stemmed = _stem(w)
        tokens.append(stemmed)
        seen_aliases.add(stemmed)

        for source in (w, stemmed):
            for alias in ALIAS_MAP.get(source, ()):
                a_stem = _stem(alias)
                if a_stem not in seen_aliases:
                    tokens.append(a_stem)
                    seen_aliases.add(a_stem)

    return tokens


# ---------------------------------------------------------------------------
# BM25 Okapi Configuration
# ---------------------------------------------------------------------------

BM25_K1 = 1.5
BM25_B = 0.75

# Hierarchical field weights — PageIndex insight: broad fields (WHAT the task
# is) carry more weight than specific fields (WHAT was done in it).
COARSE_FIELDS = {"title", "objective", "tags"}

FIELD_WEIGHTS = {
    "title": 5.0,
    "objective": 4.0,
    "tags": 4.5,
    "decisions_problem": 3.5,
    "decisions_chosen": 3.0,
    "decisions_alternatives": 2.5,
    "decisions_reasoning": 2.0,
    "current_state": 2.0,
    "next_steps": 2.0,
    "blockers": 1.5,
    "snapshots": 1.5,
    "files_changed": 2.0,
    "git_branches": 1.5,
    "bigrams": 6.0,
}

# PageIndex tree-search insight: if the same query term appears in BOTH coarse
# (title/objective) AND deep (decisions/snapshots) fields, that's strong
# multi-level confirmation of relevance.
MULTI_LEVEL_BOOST_PER_TERM = 0.15


# ---------------------------------------------------------------------------
# Task Document — hierarchical index (PageIndex tree analog)
# ---------------------------------------------------------------------------

@dataclass
class TaskDocument:
    """A hierarchical document built from a task for BM25 scoring.

    Mirrors PageIndex: fields at different specificity levels form a tree.
    Coarse fields (title/objective/tags) are the "root nodes" — they tell
    you WHAT the task is. Deep fields (decisions/snapshots) are "leaf nodes"
    — they tell you WHAT was done.
    """
    task_id: str
    task: Task
    fields: dict[str, str] = field(default_factory=dict)
    field_tokens: dict[str, list[str]] = field(default_factory=dict)
    total_tokens: int = 0

    def build(self, store: Store):
        """Populate all fields from the task's stored data."""
        t = self.task
        self.fields = {
            "title": t.title,
            "objective": t.objective,
            "tags": " ".join(t.tags),
            "current_state": t.current_state,
            "next_steps": t.next_steps,
            "blockers": t.blockers,
        }

        decisions = store.get_decisions(t.id)
        self.fields["decisions_problem"] = " ".join(d.problem for d in decisions)
        self.fields["decisions_chosen"] = " ".join(d.chosen for d in decisions)
        self.fields["decisions_alternatives"] = " ".join(
            " ".join(d.alternatives) for d in decisions
        )
        self.fields["decisions_reasoning"] = " ".join(
            f"{d.tradeoffs} {d.reasoning}" for d in decisions
        )

        snapshots = store.get_snapshots(t.id, limit=20)
        self.fields["snapshots"] = " ".join(s.message for s in snapshots)
        self.fields["files_changed"] = " ".join(
            " ".join(s.files_changed) for s in snapshots
        )
        self.fields["git_branches"] = " ".join(
            s.git_branch for s in snapshots if s.git_branch
        )

        for fname, text in self.fields.items():
            self.field_tokens[fname] = _tokenize(text)

        # Bigrams from high-value fields (compound term matching)
        bigram_source = (
            self.field_tokens.get("title", [])
            + self.field_tokens.get("objective", [])
            + self.field_tokens.get("decisions_problem", [])
            + self.field_tokens.get("decisions_chosen", [])
        )
        self.field_tokens["bigrams"] = _extract_bigrams(bigram_source)

        self.total_tokens = sum(len(toks) for toks in self.field_tokens.values())


# ---------------------------------------------------------------------------
# BM25 Relevance Engine — with hierarchical scoring
# ---------------------------------------------------------------------------

class BM25RelevanceEngine:
    """BM25 Okapi scoring with PageIndex-inspired hierarchical matching.

    Unlike flat cosine similarity, this engine:
    1. Weights term rarity (IDF) — rare terms are strong signals
    2. Weights field hierarchy — title matches > snapshot matches
    3. Boosts multi-level confirmation — matching in BOTH coarse and deep
       fields is stronger than matching in either alone
    4. Applies recency decay — recent tasks score higher
    """

    def __init__(self):
        self.documents: list[TaskDocument] = []
        self.avg_doc_len: float = 0.0
        self.doc_freq: dict[str, int] = {}
        self.n_docs: int = 0

    def index(self, store: Store):
        """Build the index from all tasks in the store."""
        all_tasks = store.list_tasks(project_only=False)
        self.documents = []

        for task in all_tasks:
            doc = TaskDocument(task_id=task.id, task=task)
            doc.build(store)
            self.documents.append(doc)

        self.n_docs = len(self.documents)
        if self.n_docs == 0:
            return

        total_len = sum(d.total_tokens for d in self.documents)
        self.avg_doc_len = total_len / self.n_docs if self.n_docs > 0 else 1.0

        self.doc_freq = {}
        for doc in self.documents:
            seen_terms = set()
            for tokens in doc.field_tokens.values():
                for t in tokens:
                    seen_terms.add(t)
            for t in seen_terms:
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for relevant tasks using hierarchical BM25 scoring.

        Returns ranked list of {task, score, confidence, evidence} dicts.
        """
        query_tokens = _tokenize(query)
        query_bigrams = _extract_bigrams(query_tokens)

        if not query_tokens:
            return []

        id_match = re.search(r"\b([a-f0-9]{8,12})\b", query.lower())

        results = []
        for doc in self.documents:
            if id_match and id_match.group(1) in doc.task_id:
                results.append({
                    "task": doc.task,
                    "score": 1000.0,
                    "confidence": 1.0,
                    "evidence": ["exact_task_id_match"],
                    "field_scores": {},
                })
                continue

            total_score = 0.0
            evidence = []
            field_scores = {}
            coarse_matched: set[str] = set()
            deep_matched: set[str] = set()

            for fname, tokens in doc.field_tokens.items():
                if not tokens:
                    continue

                weight = FIELD_WEIGHTS.get(fname, 1.0)
                field_score = 0.0

                q_terms = query_bigrams if fname == "bigrams" else query_tokens

                for qt in q_terms:
                    tf = tokens.count(qt)
                    if tf == 0:
                        continue

                    n_containing = self.doc_freq.get(qt, 0)
                    idf = math.log(
                        (self.n_docs - n_containing + 0.5)
                        / (n_containing + 0.5)
                        + 1.0
                    )

                    dl = len(tokens)
                    tf_norm = (tf * (BM25_K1 + 1)) / (
                        tf
                        + BM25_K1
                        * (1 - BM25_B + BM25_B * dl / max(self.avg_doc_len, 1))
                    )

                    term_score = idf * tf_norm * weight
                    field_score += term_score
                    evidence.append(f"{fname}:{qt}")

                    if fname in COARSE_FIELDS:
                        coarse_matched.add(qt)
                    else:
                        deep_matched.add(qt)

                if field_score > 0:
                    field_scores[fname] = field_score
                    total_score += field_score

            # --- PageIndex-inspired multi-level confirmation boost ---
            # If the same term matches in BOTH coarse (title/obj) AND deep
            # (decisions/snapshots) fields, that's strong hierarchical evidence.
            multi_level_terms = coarse_matched & deep_matched
            if multi_level_terms:
                total_score *= 1.0 + MULTI_LEVEL_BOOST_PER_TERM * len(
                    multi_level_terms
                )

            # Active task boost
            if doc.task.status == "active":
                total_score *= 1.15

            # Project-local boost
            project_path = str(Path.cwd().resolve())
            if doc.task.project_path and project_path in doc.task.project_path:
                total_score *= 1.25

            # Recency decay
            total_score *= _time_decay_factor(doc.task.updated_at)

            if total_score > 0:
                matched_terms = set()
                for e in evidence:
                    if ":" in e:
                        matched_terms.add(e.split(":")[1])

                all_query_terms = set(query_tokens + query_bigrams)

                raw_fraction = len(matched_terms) / max(len(all_query_terms), 1)

                # IDF-weighted coverage: rare term matches count for more
                # than common ones. A single match on "eucatur" (unique,
                # high-IDF) is stronger evidence than matching 3 generic terms.
                matched_idf = 0.0
                for t in matched_terms:
                    n = self.doc_freq.get(t, 0)
                    matched_idf += max(
                        math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0),
                        0.1,
                    )

                denom_idf = 0.0
                for t in all_query_terms:
                    n = self.doc_freq.get(t, 0)
                    if n > 0:
                        denom_idf += max(
                            math.log(
                                (self.n_docs - n + 0.5) / (n + 0.5) + 1.0
                            ),
                            0.1,
                        )
                    else:
                        # Term absent from corpus — minimal penalty
                        # (query noise, not negative evidence)
                        denom_idf += 0.2

                idf_fraction = (
                    min(matched_idf / max(denom_idf, 0.01), 1.0)
                    if denom_idf > 0
                    else raw_fraction
                )

                effective_fraction = max(raw_fraction, idf_fraction)

                # PageIndex tree-root boost: query-length-adaptive coarse floor.
                #
                # A term matching in a coarse field (title/objective/tags) is a
                # task-identity signal — those fields describe WHAT the task IS,
                # not just what happened inside it. Without any boost, a verbose
                # query ("I was looking at logs last week and noticed something
                # weird with eucatur...") dilutes raw_fraction (1/15 ≈ 0.07),
                # hiding a genuine specific-term match.
                #
                # DESIGN: floor = min(1.0, coarse_unigram_fraction + COARSE_BOOST)
                #
                # We use UNIGRAM coverage (not all_query_terms which includes
                # bigrams) so that a 2-word query like "check eucatur" isn't
                # penalized for generating a third "check_eucatur" bigram term.
                # Bigrams remain in the main score; the floor uses cleaner signal.
                #
                # This is query-length adaptive:
                #   Short query  ("check eucatur"), 1/2 match  → floor = 0.5+0.3 = 0.8 ✓
                #   Medium query (5 words), 1/5 match          → floor = 0.2+0.3 = 0.5
                #   Long query   (12 words), 1/12 match        → floor = 0.08+0.3 = 0.38
                #
                # Paired with the ambiguous-intent threshold (0.65): short focused
                # queries auto-resume; long verbose queries with one random match
                # stay below 0.65 and create a new task.
                COARSE_BOOST = 0.3
                if coarse_matched:
                    n_query_unigrams = max(len(set(query_tokens)), 1)
                    coarse_unigram_fraction = len(coarse_matched) / n_query_unigrams
                    effective_fraction = max(
                        effective_fraction,
                        min(1.0, coarse_unigram_fraction + COARSE_BOOST),
                    )

                results.append({
                    "task": doc.task,
                    "score": total_score,
                    "confidence": 0.0,
                    "evidence": evidence[:10],
                    "field_scores": field_scores,
                    "_matched_fraction": effective_fraction,
                })

        if not results:
            return []

        max_score = max(r["score"] for r in results)
        for r in results:
            score_norm = (
                min(r["score"] / max_score, 1.0) if max_score > 0 else 0
            )
            r["confidence"] = score_norm * r.pop("_matched_fraction")

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]


# ---------------------------------------------------------------------------
# Resume Briefing Generator
# ---------------------------------------------------------------------------

def generate_resume_briefing(task_id: str, store: Store) -> str:
    """Generate a comprehensive briefing for an agent resuming a task.

    This is NOT just a handoff bundle. It is structured to prevent the
    resuming agent from producing garbage by explicitly surfacing:
    1. WARNINGS — dead ends, failed experiments, things NOT to do
    2. ARCHITECTURE — decisions with reasoning (don't change these)
    3. EXACT STATE — what files were touched, what tests pass
    4. RESUMPTION POINT — where exactly to pick up
    5. VERIFICATION — commands to run to confirm understanding
    """
    task = store.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    decisions = store.get_decisions(task_id)
    snapshots = store.get_snapshots(task_id, limit=20)

    lines = [
        "# Stitch Resume Briefing",
        "",
        f"**Task**: {task.title} (`{task.id}`)",
        f"**Project**: `{task.project_path}`",
        f"**Status**: {task.status}",
        "",
    ]

    # --- Section 1: WARNINGS (most critical — read first) ---
    warnings = []
    for s in snapshots:
        msg = s.message.upper()
        if any(w in msg for w in ["FAIL", "DEAD", "ERROR", "BROKEN", "REVERT", "ABORT"]):
            warnings.append(f"- [{s.timestamp}] {s.message}")
    for s in snapshots:
        if s.extra and s.extra.get("failures"):
            warnings.append(f"- DEAD ENDS: {s.extra['failures']}")

    if warnings:
        lines += [
            "## ⚠ WARNINGS — Do NOT Repeat These Mistakes",
            "",
            "Previous sessions encountered these failures. Do NOT try these approaches again",
            "without understanding why they failed:",
            "",
        ] + warnings + [""]

    # --- Section 2: Architecture Decisions (DO NOT change without reason) ---
    if decisions:
        lines += [
            "## Architecture Decisions (DO NOT override without strong reason)",
            "",
        ]
        for d in decisions:
            alts = ", ".join(d.alternatives) if d.alternatives else "(none recorded)"
            lines += [
                f"### {d.problem}",
                f"**Chosen**: {d.chosen}",
                f"**Rejected alternatives**: {alts}",
                f"**Reasoning**: {d.reasoning}",
            ]
            if d.tradeoffs:
                lines.append(f"**Tradeoffs accepted**: {d.tradeoffs}")
            lines.append("")

    # --- Section 3: Objective & Current State ---
    lines += [
        "## Objective",
        task.objective or "(not set)",
        "",
        "## Current State",
        task.current_state or "(not set)",
        "",
    ]

    # --- Section 4: Exact Resumption Point ---
    lines += [
        "## Exact Resumption Point",
        "",
        "**Next steps (do these in order)**:",
        task.next_steps or "(not set)",
        "",
    ]

    if task.blockers:
        lines += [f"**Blockers**: {task.blockers}", ""]

    # --- Section 5: Recent Activity (chronological) ---
    if snapshots:
        lines += ["## Session History (most recent first)", ""]
        for s in snapshots[-10:]:
            src_tag = f" [{s.source}]" if s.source != "manual" else ""
            lines.append(f"- **{s.timestamp}**{src_tag}: {s.message[:200]}")
        lines.append("")

    # --- Section 6: Files & Branches ---
    all_files = set()
    branches = set()
    for s in snapshots:
        all_files.update(s.files_changed[:10])
        if s.git_branch:
            branches.add(s.git_branch)

    if all_files:
        lines += ["## Files Touched", ""]
        for f in sorted(all_files)[:20]:
            lines.append(f"- `{f}`")
        lines.append("")

    if branches:
        lines += [
            "## Git Branches Used",
            "",
            ", ".join(f"`{b}`" for b in sorted(branches)),
            "",
        ]

    # --- Section 7: Verification Instructions ---
    lines += [
        "## Verification (run these BEFORE writing code)",
        "",
        "1. Read the files listed above to understand the current codebase state",
        "2. Run tests if they exist to confirm baseline",
        "3. Check `git status` and `git log --oneline -5` to verify branch and recent commits",
        "4. Confirm the Architecture Decisions above still make sense for the objective",
        "5. Only THEN proceed with the Next Steps",
        "",
    ]

    # --- Section 8: Project Repo Validation ---
    project_path = task.project_path
    if project_path and os.path.isdir(project_path):
        lines += ["## Project Repo State (live)", ""]
        if is_git_repo(project_path):
            branch = run_git(["branch", "--show-current"], cwd=project_path)
            status = run_git(["status", "--short"], cwd=project_path)
            last_commit = run_git(["log", "-1", "--format=%h %s (%ar)"], cwd=project_path)
            lines.append(f"**Branch**: `{branch}`")
            lines.append(f"**Last commit**: `{last_commit}`")
            if status.strip():
                lines.append(f"**Uncommitted changes**: {len(status.strip().splitlines())} files")
            else:
                lines.append("**Working tree**: clean")
        else:
            lines.append("(not a git repo)")
        lines.append("")
    elif project_path:
        lines += [
            "## ⚠ Project Repo Warning",
            f"Project path `{project_path}` does not exist on this machine.",
            "The task may have been created on a different machine or the directory was moved.",
            "",
        ]

    lines += [
        "---",
        f"*Resume briefing generated by Stitch v0.2.0 for task `{task.id}`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workspace Scanner
# ---------------------------------------------------------------------------

def _check_stitch_project_data(entry: Path, query_tokens: list[str]) -> tuple[float, list[str]]:
    """Check ~/.stitch/projects/ for task data belonging to a workspace entry."""
    from .store import PROJECTS_HOME, project_key
    score = 0.0
    evidence = []

    key = project_key(entry)
    proj_dir = PROJECTS_HOME / key

    if proj_dir.is_dir():
        score += 2.0
        evidence.append("has_stitch")

        active_file = proj_dir / "active_task"
        if active_file.exists():
            try:
                task_id = active_file.read_text().strip()
                meta_file = proj_dir / "tasks" / task_id / "meta.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    title = meta.get("title", "").lower()
                    for qt in query_tokens:
                        if qt in title:
                            score += 4.0
                            evidence.append(f"task_title:{qt}")
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    # Backward compat: also check old in-repo .stitch/
    old_stitch = entry / ".stitch"
    if old_stitch.is_dir() and score == 0:
        score += 2.0
        evidence.append("has_stitch_legacy")
        active_file = old_stitch / "active_task"
        if active_file.exists():
            try:
                task_id = active_file.read_text().strip()
                meta_file = old_stitch / "tasks" / task_id / "meta.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    title = meta.get("title", "").lower()
                    for qt in query_tokens:
                        if qt in title:
                            score += 4.0
                            evidence.append(f"task_title:{qt}")
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    return score, evidence


def scan_workspace_for_context(
    workspace_root: str,
    query_tokens: list[str],
) -> list[dict]:
    """Scan local workspace projects for additional context signals.

    Looks at project directories under the workspace root to find
    repos that might be related to a query (by checking directory names,
    README files, and recent git log messages).
    Checks ~/.stitch/projects/ for task data (new layout) and falls back
    to in-repo .stitch/ (legacy layout).
    """
    root = Path(workspace_root)
    if not root.is_dir():
        return []

    hints = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        score = 0
        evidence = []
        name_lower = entry.name.lower().replace("-", " ").replace("_", " ")

        for qt in query_tokens:
            if qt in name_lower:
                score += 3.0
                evidence.append(f"dirname:{qt}")

        stitch_score, stitch_evidence = _check_stitch_project_data(entry, query_tokens)
        score += stitch_score
        evidence.extend(stitch_evidence)

        if score > 0 and is_git_repo(str(entry)):
            recent_log = run_git(["log", "--oneline", "-5"], cwd=str(entry))
            log_lower = recent_log.lower()
            for qt in query_tokens:
                if qt in log_lower:
                    score += 1.0
                    evidence.append(f"git_log:{qt}")

        if score > 0:
            hints.append({
                "project_path": str(entry),
                "project_name": entry.name,
                "score": score,
                "evidence": evidence,
            })

    hints.sort(key=lambda x: -x["score"])
    return hints[:5]
