#!/usr/bin/env python3
"""Build ClinicalTrials.gov probe corpora, CKGs, and benchmark queries.

This is a deterministic expansion scaffold for Track 3. It pulls public
ClinicalTrials.gov API v2 records for a manifest of therapeutic areas, writes a
markdown corpus for RAG/GraphRAG, derives a compact learning-graph.csv, and
generates benchmark queries using the repo's existing query generator.

No LLM is used to build the CKGs. Nodes are extracted from trial metadata:
conditions, interventions, outcomes, sponsors, phases, statuses, and selected
NCT trial records. Provenance is stored in sidecar JSON so the benchmark CSV
schema remains compatible with the existing harnesses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmark" / "clinicaltrials" / "top12_domains.json"
USER_AGENT = "ckg-benchmark-clinicaltrials-probe/2026-09-08"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_queries import generate_queries, load_csv  # noqa: E402


@dataclass
class Node:
    concept_id: int
    label: str
    dependencies: list[int]
    taxonomy_id: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_label(value: Any, limit: int = 120) -> str:
    text = clean_text(value)
    text = re.sub(r"^[*-]\s+", "", text)
    text = text.strip(" .;:,")
    if len(text) > limit:
        text = text[: limit - 3].rstrip(" .;:,") + "..."
    return text


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "item"


def truncate(value: str, limit: int = 2200) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def first_sentence(value: str, limit: int = 140) -> str:
    text = clean_text(value)
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s+", text)
    if match:
        text = text[: match.start()]
    return clean_label(text, limit)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def source_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_trials_for_query(
    api_base: str,
    query_spec: dict[str, str],
    max_trials: int,
    page_size: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    next_page: str | None = None
    page_size = max(1, min(page_size, 100))

    while len(trials) < max_trials:
        params = {
            query_spec["field"]: query_spec["term"],
            "format": "json",
            "pageSize": str(min(page_size, max_trials - len(trials))),
        }
        if next_page:
            params["pageToken"] = next_page

        url = api_base + "?" + urllib.parse.urlencode(params)
        data = fetch_json(url)
        studies = as_list(data.get("studies"))
        if not studies:
            break

        for study in studies:
            trial = extract_trial(study, search_term=query_spec["term"])
            if trial.get("nct_id"):
                trials.append(trial)

        next_page = data.get("nextPageToken")
        if not next_page or len(studies) < page_size:
            break
        time.sleep(sleep_seconds)

    return trials


def outcome_records(outcomes: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = []
    for outcome in outcomes:
        measure = clean_label(outcome.get("measure") or outcome.get("description"), 140)
        description = clean_text(outcome.get("description"))
        if measure:
            records.append({"measure": measure, "description": description})
    return records


def intervention_records(interventions: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = []
    for intervention in interventions:
        name = clean_label(intervention.get("name"), 120)
        if not name:
            continue
        records.append(
            {
                "name": name,
                "type": clean_label(intervention.get("type"), 80),
                "description": clean_text(intervention.get("description")),
            }
        )
    return records


def extract_trial(study: dict[str, Any], search_term: str) -> dict[str, Any]:
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    desc_mod = proto.get("descriptionModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    status_mod = proto.get("statusModule", {})
    design_mod = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    conditions_mod = proto.get("conditionsModule", {})

    nct_id = clean_label(id_mod.get("nctId"), 32)
    primary_outcomes = outcome_records(as_list(outcomes_mod.get("primaryOutcomes")))
    secondary_outcomes = outcome_records(as_list(outcomes_mod.get("secondaryOutcomes")))
    lead_sponsor = sponsor_mod.get("leadSponsor") or {}

    return {
        "nct_id": nct_id,
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "title": clean_text(id_mod.get("briefTitle")),
        "official_title": clean_text(id_mod.get("officialTitle")),
        "brief_summary": clean_text(desc_mod.get("briefSummary")),
        "detailed_description": clean_text(desc_mod.get("detailedDescription")),
        "eligibility_criteria": clean_text(elig_mod.get("eligibilityCriteria")),
        "conditions": [clean_label(c, 120) for c in as_list(conditions_mod.get("conditions")) if clean_label(c)],
        "interventions": intervention_records(as_list(arms_mod.get("interventions"))),
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "phases": [clean_label(p, 60) for p in as_list(design_mod.get("phases")) if clean_label(p)],
        "status": clean_label(status_mod.get("overallStatus"), 80),
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count", ""),
        "sponsor": clean_label(lead_sponsor.get("name"), 140),
        "study_type": clean_label(design_mod.get("studyType"), 80),
        "search_terms": [search_term],
        "source_hash": source_hash(study),
    }


def merge_trial(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing_terms = list(existing.get("search_terms", []))
    for term in incoming.get("search_terms", []):
        if term not in existing_terms:
            existing_terms.append(term)
    existing["search_terms"] = existing_terms


def fetch_domain_trials(
    manifest: dict[str, Any],
    domain: dict[str, Any],
    max_trials: int,
    page_size: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    by_nct: dict[str, dict[str, Any]] = {}
    queries = as_list(domain.get("queries"))
    per_query_limit = max(1, (max_trials + max(len(queries), 1) - 1) // max(len(queries), 1))

    for query_spec in queries:
        term = query_spec.get("term", "")
        field = query_spec.get("field", "")
        print(f"  fetching {field}={term!r} up to {per_query_limit} trials")
        trials = fetch_trials_for_query(
            api_base=manifest["api_base"],
            query_spec=query_spec,
            max_trials=per_query_limit,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
        )
        for trial in trials:
            nct_id = trial["nct_id"]
            if nct_id in by_nct:
                merge_trial(by_nct[nct_id], trial)
            else:
                by_nct[nct_id] = trial
        print(f"    unique domain trials so far: {len(by_nct)}")

    trials = list(by_nct.values())
    trials.sort(key=lambda item: item.get("nct_id", ""))
    return trials[:max_trials]


def trial_to_markdown(trial: dict[str, Any]) -> str:
    interventions = "; ".join(
        f"{item['type']}: {item['name']}" if item.get("type") else item["name"]
        for item in trial.get("interventions", [])
    )
    primary = "; ".join(item["measure"] for item in trial.get("primary_outcomes", []))
    secondary = "; ".join(item["measure"] for item in trial.get("secondary_outcomes", [])[:8])
    search_terms = ", ".join(trial.get("search_terms", []))

    parts = [
        f"## {trial['nct_id']}: {trial.get('title', '')}",
        f"ClinicalTrials.gov: {trial.get('url', '')}",
        f"Matched search terms: {search_terms}",
        (
            f"Status: {trial.get('status', '')} | Phase: {', '.join(trial.get('phases', []))} | "
            f"Enrollment: {trial.get('enrollment', '')} | Sponsor: {trial.get('sponsor', '')}"
        ),
        f"Conditions: {', '.join(trial.get('conditions', []))}",
    ]
    if interventions:
        parts.append(f"Interventions: {interventions}")
    if primary:
        parts.append(f"Primary outcomes: {primary}")
    if secondary:
        parts.append(f"Secondary outcomes: {secondary}")
    if trial.get("brief_summary"):
        parts.append("Summary: " + truncate(trial["brief_summary"]))
    if trial.get("detailed_description"):
        parts.append("Detailed description: " + truncate(trial["detailed_description"]))
    if trial.get("eligibility_criteria"):
        parts.append("Eligibility criteria: " + truncate(trial["eligibility_criteria"]))
    return "\n\n".join(parts)


def write_corpus_docs(
    root: Path,
    domain: dict[str, Any],
    trials: list[dict[str, Any]],
    force: bool,
) -> dict[str, Any]:
    domain_id = domain["id"]
    docs_dir = root / "corpus" / domain_id / "docs"
    chapters_dir = docs_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    by_term: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        for term in trial.get("search_terms", []):
            by_term[term].append(trial)

    course_description = [
        f"# {domain['label']} ClinicalTrials.gov Corpus",
        "",
        "Public ClinicalTrials.gov API v2 records normalized for CKG/RAG/GraphRAG probe testing.",
        "",
        f"Domain id: {domain_id}",
        f"Rationale: {domain.get('rationale', '')}",
        "",
        "Search terms:",
    ]
    for query_spec in domain.get("queries", []):
        course_description.append(f"- {query_spec['field']}={query_spec['term']}")
    (docs_dir / "course-description.md").write_text("\n".join(course_description) + "\n")

    chapter_count = 0
    for index, query_spec in enumerate(domain.get("queries", []), start=1):
        term = query_spec["term"]
        chapter_trials = by_term.get(term, [])
        chapter_slug = f"{index:02d}-{slugify(term)}"
        chapter_dir = chapters_dir / chapter_slug
        chapter_dir.mkdir(parents=True, exist_ok=True)
        chapter_path = chapter_dir / "index.md"
        lines = [
            f"# {domain['label']}: {term}",
            "",
            f"Source: ClinicalTrials.gov API v2, {query_spec['field']}={term}",
            f"Trial count in this chapter: {len(chapter_trials)}",
            "",
        ]
        for trial in chapter_trials:
            lines.append(trial_to_markdown(trial))
            lines.append("")
        chapter_path.write_text("\n".join(lines).strip() + "\n")
        chapter_count += 1

    return {"docs_dir": str(docs_dir), "chapters": chapter_count}


def top_labels(counter: Counter[str], limit: int) -> list[str]:
    labels = []
    for label, _count in counter.most_common():
        if label and label.lower() not in {"unknown", "not applicable", "na", "n/a"}:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def ranked_labels(counter: Counter[str]) -> list[str]:
    return top_labels(counter, len(counter))


def build_graph(domain: dict[str, Any], trials: list[dict[str, Any]], max_concepts: int) -> tuple[list[Node], dict[str, Any]]:
    nodes: list[Node] = []
    index_by_key: dict[tuple[str, str], int] = {}
    provenance_by_key: defaultdict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    def add_node(label: str, taxonomy_id: str, dependencies: list[int] | None = None) -> int:
        label = clean_label(label)
        if not label:
            label = "Unknown"
        key = (taxonomy_id, label.lower())
        deps = list(dict.fromkeys(dependencies or []))
        if key in index_by_key:
            node = nodes[index_by_key[key] - 1]
            merged = list(dict.fromkeys(node.dependencies + deps))
            node.dependencies = merged
            return node.concept_id
        concept_id = len(nodes) + 1
        index_by_key[key] = concept_id
        nodes.append(Node(concept_id, label, deps, taxonomy_id))
        return concept_id

    root = add_node(f"{domain['label']} clinical trial landscape", "AREA")
    condition_root = add_node("Condition landscape", "FRAME", [root])
    intervention_root = add_node("Intervention landscape", "FRAME", [root])
    endpoint_root = add_node("Endpoint landscape", "FRAME", [root])
    sponsor_root = add_node("Sponsor landscape", "FRAME", [root])
    design_root = add_node("Trial design landscape", "FRAME", [root])
    status_root = add_node("Recruitment and status landscape", "FRAME", [root])

    condition_counts: Counter[str] = Counter()
    intervention_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    sponsor_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for trial in trials:
        source_pair = (trial.get("url", ""), trial.get("source_hash", ""))
        for condition in trial.get("conditions", []):
            condition_counts[condition] += 1
            provenance_by_key[("COND", condition.lower())].add(source_pair)
        for item in trial.get("interventions", []):
            name = item.get("name", "")
            if name:
                intervention_counts[name] += 1
                provenance_by_key[("INTR", name.lower())].add(source_pair)
        for outcome in trial.get("primary_outcomes", []) + trial.get("secondary_outcomes", []):
            label = outcome.get("measure") or first_sentence(outcome.get("description", ""))
            if label:
                outcome_counts[label] += 1
                provenance_by_key[("OUTC", label.lower())].add(source_pair)
        if trial.get("sponsor"):
            sponsor_counts[trial["sponsor"]] += 1
            provenance_by_key[("SPON", trial["sponsor"].lower())].add(source_pair)
        phases = trial.get("phases", []) or ["Not Applicable"]
        for phase in phases:
            phase_counts[phase] += 1
            provenance_by_key[("PHASE", phase.lower())].add(source_pair)
        if trial.get("status"):
            status_counts[trial["status"]] += 1
            provenance_by_key[("STATUS", trial["status"].lower())].add(source_pair)

    used_labels = {node.label.lower() for node in nodes}

    def add_ranked_nodes(
        counter: Counter[str],
        taxonomy_id: str,
        dependency_id: int,
        limit: int,
    ) -> dict[str, int]:
        ids = {}
        for label in ranked_labels(counter):
            if len(ids) >= limit or len(nodes) >= max_concepts:
                break
            label_key = label.lower()
            if label_key in used_labels:
                continue
            ids[label] = add_node(label, taxonomy_id, [dependency_id])
            used_labels.add(label_key)
        return ids

    condition_ids = add_ranked_nodes(condition_counts, "COND", condition_root, 30)
    intervention_ids = add_ranked_nodes(intervention_counts, "INTR", intervention_root, 40)
    outcome_ids = add_ranked_nodes(outcome_counts, "OUTC", endpoint_root, 35)
    sponsor_ids = add_ranked_nodes(sponsor_counts, "SPON", sponsor_root, 25)
    phase_ids = add_ranked_nodes(phase_counts, "PHASE", design_root, 12)
    status_ids = add_ranked_nodes(status_counts, "STATUS", status_root, 12)

    ranked_trials = sorted(
        trials,
        key=lambda trial: (
            len(trial.get("conditions", []))
            + len(trial.get("interventions", []))
            + len(trial.get("primary_outcomes", [])),
            trial.get("nct_id", ""),
        ),
        reverse=True,
    )
    for trial in ranked_trials:
        if len(nodes) >= max_concepts:
            break
        deps = []
        for condition in trial.get("conditions", [])[:2]:
            if condition in condition_ids:
                deps.append(condition_ids[condition])
        for item in trial.get("interventions", [])[:2]:
            if item.get("name") in intervention_ids:
                deps.append(intervention_ids[item["name"]])
        for outcome in trial.get("primary_outcomes", [])[:1]:
            label = outcome.get("measure", "")
            if label in outcome_ids:
                deps.append(outcome_ids[label])
        if trial.get("sponsor") in sponsor_ids:
            deps.append(sponsor_ids[trial["sponsor"]])
        for phase in trial.get("phases", [])[:1]:
            if phase in phase_ids:
                deps.append(phase_ids[phase])
        if trial.get("status") in status_ids:
            deps.append(status_ids[trial["status"]])
        if not deps:
            deps.append(root)
        concept_id = add_node(f"{trial['nct_id']} {trial.get('title', '')}", "TRIAL", deps)
        provenance_by_key[("TRIAL", nodes[concept_id - 1].label.lower())].add(
            (trial.get("url", ""), trial.get("source_hash", ""))
        )

    sources_by_node: dict[str, Any] = {}
    for node in nodes:
        key = (node.taxonomy_id, node.label.lower())
        pairs = sorted(provenance_by_key.get(key, set()))
        sources_by_node[str(node.concept_id)] = {
            "label": node.label,
            "taxonomy_id": node.taxonomy_id,
            "dependencies": node.dependencies,
            "sources": [
                {"url": url, "source_hash": hash_value}
                for url, hash_value in pairs[:50]
                if url or hash_value
            ],
        }

    summary = {
        "domain_id": domain["id"],
        "domain_label": domain["label"],
        "trial_count": len(trials),
        "node_count": len(nodes),
        "edge_count": sum(len(node.dependencies) for node in nodes),
        "taxonomy_counts": dict(Counter(node.taxonomy_id for node in nodes)),
        "source_coverage": (
            sum(1 for item in sources_by_node.values() if item["sources"]) / len(nodes)
            if nodes
            else 0.0
        ),
    }
    return nodes, {"summary": summary, "nodes": sources_by_node}


def write_graph(root: Path, domain_id: str, nodes: list[Node], provenance: dict[str, Any]) -> Path:
    graph_dir = root / "benchmark" / "domains" / domain_id
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "learning-graph.csv"
    with graph_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ConceptID", "ConceptLabel", "Dependencies", "TaxonomyID"])
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    "ConceptID": node.concept_id,
                    "ConceptLabel": node.label,
                    "Dependencies": "|".join(str(dep) for dep in sorted(set(node.dependencies))),
                    "TaxonomyID": node.taxonomy_id,
                }
            )
    (graph_dir / "sources.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return graph_path


def write_queries(root: Path, domain_id: str, graph_path: Path, seed: int) -> Path:
    queries_dir = root / "benchmark" / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    concepts = load_csv(str(graph_path))
    queries = generate_queries(concepts, domain_id, seed=seed)
    output_path = queries_dir / f"queries_{domain_id}.jsonl"
    with output_path.open("w") as fh:
        for query in queries:
            fh.write(json.dumps(query, sort_keys=True) + "\n")
    return output_path


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        manifest = json.load(fh)
    ids = [domain["id"] for domain in manifest.get("domains", [])]
    duplicates = [domain_id for domain_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate domain ids in manifest: {', '.join(sorted(duplicates))}")
    return manifest


def select_domains(manifest: dict[str, Any], selected: list[str] | None, all_domains: bool) -> list[dict[str, Any]]:
    domains = list(manifest.get("domains", []))
    if all_domains:
        return domains
    selected = selected or []
    by_id = {domain["id"]: domain for domain in domains}
    missing = [domain_id for domain_id in selected if domain_id not in by_id]
    if missing:
        raise ValueError(f"unknown domain ids: {', '.join(missing)}")
    return [by_id[domain_id] for domain_id in selected]


def load_cached_corpus(corpus_path: Path) -> list[dict[str, Any]] | None:
    if not corpus_path.exists():
        return None
    with corpus_path.open() as fh:
        return json.load(fh)


def write_domain_artifacts(
    manifest: dict[str, Any],
    domain: dict[str, Any],
    max_trials: int,
    max_concepts: int,
    page_size: int,
    sleep_seconds: float,
    seed: int,
    refresh: bool,
    dry_run: bool,
) -> dict[str, Any]:
    domain_id = domain["id"]
    results_dir = PROJECT_ROOT / "results" / "clinicaltrials" / domain_id
    corpus_path = results_dir / "corpus.json"
    summary_path = results_dir / "summary.json"
    graph_path = PROJECT_ROOT / "benchmark" / "domains" / domain_id / "learning-graph.csv"
    query_path = PROJECT_ROOT / "benchmark" / "queries" / f"queries_{domain_id}.jsonl"

    if dry_run:
        return {
            "domain_id": domain_id,
            "label": domain["label"],
            "would_fetch": [f"{item['field']}={item['term']}" for item in domain.get("queries", [])],
            "would_write": [
                str(corpus_path.relative_to(PROJECT_ROOT)),
                str(summary_path.relative_to(PROJECT_ROOT)),
                str(graph_path.relative_to(PROJECT_ROOT)),
                str(query_path.relative_to(PROJECT_ROOT)),
            ],
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    trials = None if refresh else load_cached_corpus(corpus_path)
    if trials is None:
        trials = fetch_domain_trials(
            manifest=manifest,
            domain=domain,
            max_trials=max_trials,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
        )
        corpus_path.write_text(json.dumps(trials, indent=2, sort_keys=True) + "\n")
    else:
        print(f"  using cached corpus: {corpus_path.relative_to(PROJECT_ROOT)}")

    docs_summary = write_corpus_docs(PROJECT_ROOT, domain, trials, force=True)
    nodes, provenance = build_graph(domain, trials, max_concepts=max_concepts)
    graph_path = write_graph(PROJECT_ROOT, domain_id, nodes, provenance)
    query_path = write_queries(PROJECT_ROOT, domain_id, graph_path, seed=seed)

    summary = {
        "domain_id": domain_id,
        "label": domain["label"],
        "source": manifest.get("source", ""),
        "api_base": manifest.get("api_base", ""),
        "queries": domain.get("queries", []),
        "trial_count": len(trials),
        "docs": docs_summary,
        "graph": provenance["summary"],
        "outputs": {
            "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "graph": str(graph_path.relative_to(PROJECT_ROOT)),
            "provenance": str((graph_path.parent / "sources.json").relative_to(PROJECT_ROOT)),
            "queries": str(query_path.relative_to(PROJECT_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--all", action="store_true", help="Build all domains in the manifest")
    selector.add_argument("--domains", nargs="+", help="One or more manifest domain ids")
    parser.add_argument("--max-trials", type=int, default=500, help="Max unique trials per domain")
    parser.add_argument("--max-concepts", type=int, default=180, help="Max CKG nodes per domain")
    parser.add_argument("--page-size", type=int, default=100, help="ClinicalTrials.gov API page size")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between paginated API calls")
    parser.add_argument("--seed", type=int, default=42, help="Benchmark query generation seed")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached corpus JSON and fetch again")
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without fetching or writing")
    args = parser.parse_args()

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = load_manifest(manifest_path)
    domains = select_domains(manifest, selected=args.domains, all_domains=args.all)

    print(f"ClinicalTrials.gov probe builder")
    print(f"manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"domains: {len(domains)}")
    print(f"max_trials/domain: {args.max_trials}")
    print(f"max_concepts/domain: {args.max_concepts}")

    outputs = []
    failures = []
    for index, domain in enumerate(domains, start=1):
        print(f"\n[{index}/{len(domains)}] {domain['id']} - {domain['label']}")
        try:
            output = write_domain_artifacts(
                manifest=manifest,
                domain=domain,
                max_trials=args.max_trials,
                max_concepts=args.max_concepts,
                page_size=args.page_size,
                sleep_seconds=args.sleep,
                seed=args.seed,
                refresh=args.refresh,
                dry_run=args.dry_run,
            )
            outputs.append(output)
            if args.dry_run:
                for path in output["would_write"]:
                    print(f"  would write: {path}")
            else:
                graph = output["graph"]
                print(
                    "  wrote "
                    f"{graph['node_count']} nodes, {graph['edge_count']} edges, "
                    f"{output['trial_count']} trials"
                )
                print(f"  graph: {output['outputs']['graph']}")
                print(f"  queries: {output['outputs']['queries']}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
            failures.append({"domain_id": domain["id"], "error": str(exc)})
            print(f"  ERROR: {exc}")

    if not args.dry_run:
        rollup_path = PROJECT_ROOT / "results" / "clinicaltrials" / "top12_probe_summary.json"
        rollup_path.parent.mkdir(parents=True, exist_ok=True)
        rollup = {
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "domain_count": len(outputs),
            "failure_count": len(failures),
            "domains": outputs,
            "failures": failures,
        }
        rollup_path.write_text(json.dumps(rollup, indent=2, sort_keys=True) + "\n")
        print(f"\nrollup: {rollup_path.relative_to(PROJECT_ROOT)}")

    if failures:
        print(f"\ncompleted with {len(failures)} failure(s)")
        return 1

    print("\ncomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
