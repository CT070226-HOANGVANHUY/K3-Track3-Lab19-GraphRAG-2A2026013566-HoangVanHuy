"""Download 5,000 HackerNoon rows, add an LLM-built graph, and evaluate.

The graph extraction intentionally remains capped at EXTRACTION_MAX_CHUNKS
(400 in the lab) so the run is reproducible and does not create an
unbounded number of provider calls.  All 5,000 source rows are still used by
the FlatRAG index and by the evaluation corpus.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

import nbformat
import pandas as pd
from datasets import load_dataset

from run_live_evaluation import load_notebook_namespace, judge_with_retry


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "hf_live_5000.csv"
GOLDEN = ROOT / "data" / "golden_dataset.csv"
OUT = ROOT / "outputs"


def download_rows(ns: dict, limit: int = 5000) -> None:
    if SOURCE.exists() and sum(1 for _ in SOURCE.open(encoding="utf-8")) - 1 >= limit:
        print(f"Source already has {limit:,} rows: {SOURCE}")
        return

    token = ns.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("Thiếu HF_TOKEN; không thể tải đủ 5.000 dòng từ Hugging Face.")

    stream = load_dataset(
        "HackerNoon/tech-company-news-data-dump",
        split="train",
        streaming=True,
        token=token,
    )
    iterator = iter(stream)
    first = next(iterator)
    headers = list(first.keys())
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(first)
        written = 1
        for row in iterator:
            writer.writerow(row)
            written += 1
            if written >= limit:
                break
    print(f"Downloaded {written:,} rows to {SOURCE}")


def extract_parallel(ns: dict, source_df: pd.DataFrame, batch_size: int = 4, workers: int = 4):
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    batches = [source_df.iloc[i : i + batch_size] for i in range(0, len(source_df), batch_size)]

    def one(batch):
        try:
            obj, _ = ns["extract_batch"](batch)
            return obj, ""
        except Exception as exc:
            return None, repr(exc)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, batch) for batch in batches]
        for index, future in enumerate(as_completed(futures), start=1):
            obj, error = future.result()
            results.append((obj, error))
            if index % 10 == 0 or index == len(futures):
                print(f"Extraction: {index}/{len(futures)} batches", flush=True)

    triples = []
    errors = []
    for obj, error in results:
        if error:
            errors.append({"error": error})
            continue
        for item in (obj or {}).get("items", []):
            chunk_id = item.get("chunk_id")
            if chunk_id not in meta:
                continue
            for relation in item.get("relations", []):
                source = ns["norm_space"](relation.get("source"))
                target = ns["norm_space"](relation.get("target"))
                source_type = relation.get("source_type")
                target_type = relation.get("target_type")
                rel = relation.get("relation")
                if not source or not target:
                    continue
                if source_type not in ns["ALLOWED_NODE_TYPES"] or target_type not in ns["ALLOWED_NODE_TYPES"]:
                    continue
                if rel not in ns["ALLOWED_RELATIONS"]:
                    continue
                triples.append({
                    "source_raw": source,
                    "source_type": source_type,
                    "relation": rel,
                    "target_raw": target,
                    "target_type": target_type,
                    "source_chunk_id": chunk_id,
                    "published_date": meta[chunk_id] or "",
                    "evidence": ns["norm_space"](relation.get("evidence")),
                    "confidence": float(relation.get("confidence") or 0.0),
                })
    return pd.DataFrame(triples), pd.DataFrame(errors)


def evaluate(ns: dict, chunks: pd.DataFrame) -> pd.DataFrame:
    ns["build_flat_index"](chunks)
    nodes = pd.DataFrame(ns["run_cypher"](
        "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, "
        "n.name_norm AS name_norm, n.entity_type AS type, "
        "coalesce(n.aliases, []) AS aliases, coalesce(n.aliases_norm, []) AS aliases_norm"
    ))
    ns["build_entity_matcher"](nodes)
    golden = pd.read_csv(GOLDEN).fillna("")

    def retrieve(row):
        flat_context, _ = ns["retrieve_flat_context"](row.question, k=6)
        debug = ns["retrieve_graph_context"](row.question, max_hops=2, edge_limit=50, return_debug=True)
        graph_context = f"=== GRAPH ===\n{debug['context']}\n\n=== VECTOR ===\n{flat_context}"
        return row, flat_context, graph_context, debug

    def generate(item):
        row, flat_context, graph_context, debug = item
        return row, flat_context, graph_context, debug, ns["generate_answer"](row.question, flat_context), ns["generate_answer"](row.question, graph_context)

    def score(item):
        row, flat_context, graph_context, debug, flat, graph = item
        flat_judge = judge_with_retry(ns, row.question, row.reference_answer, flat["answer"], flat_context)
        graph_judge = judge_with_retry(ns, row.question, row.reference_answer, graph["answer"], graph_context)
        return {
            "id": row.id, "group": row.group, "question": row.question,
            "reference_answer": row.reference_answer,
            "flat_answer": flat["answer"], "graph_answer": graph["answer"],
            "flat_comprehensiveness": flat_judge["comprehensiveness"],
            "graph_comprehensiveness": graph_judge["comprehensiveness"],
            "flat_faithfulness": flat_judge["faithfulness"],
            "graph_faithfulness": graph_judge["faithfulness"],
            "flat_multi_hop_reasoning": flat_judge["multi_hop_reasoning"],
            "graph_multi_hop_reasoning": graph_judge["multi_hop_reasoning"],
            "flat_latency_s": flat["latency_s"], "graph_latency_s": graph["latency_s"],
            "flat_total_tokens": flat.get("total_tokens"), "graph_total_tokens": graph.get("total_tokens"),
            "flat_judge_rationale": flat_judge["rationale"],
            "graph_judge_rationale": graph_judge["rationale"],
            "graph_supernode_events": len(debug["diagnostics"].get("supernode_events", [])),
        }

    rows = list(golden.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=4) as pool:
        retrieved = list(pool.map(retrieve, rows))
        generated = list(pool.map(generate, retrieved))

    scored = []
    checkpoint = OUT / "graphrag_eval_actual_checkpoint_5000.csv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(score, item) for item in generated]
        for i, future in enumerate(as_completed(futures), start=1):
            scored.append(future.result())
            pd.DataFrame(scored).sort_values("id").to_csv(checkpoint, index=False)
            if i % 5 == 0 or i == len(futures):
                print(f"Evaluation: {i}/{len(futures)} questions", flush=True)

    result = pd.DataFrame(scored).sort_values("id").reset_index(drop=True)
    metrics = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }
    summary = []
    for group, group_df in result.groupby("group", sort=True):
        for metric, (flat_col, graph_col) in metrics.items():
            flat = float(pd.to_numeric(group_df[flat_col], errors="coerce").mean())
            graph = float(pd.to_numeric(group_df[graph_col], errors="coerce").mean())
            summary.append({"Loại câu hỏi": group, "Metric": metric,
                            "Flat RAG": round(flat, 3), "GraphRAG": round(graph, 3),
                            "Delta (Graph - Flat)": round(graph - flat, 3)})
    result.to_csv(OUT / "graphrag_eval_results_5000.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "graphrag_vs_flatrag_summary_5000.csv", index=False)
    return result, pd.DataFrame(summary)


def main() -> None:
    ns = load_notebook_namespace()
    # Keep each provider request bounded; the notebook's interactive client
    # uses a much longer default timeout, which is unsuitable for a batch run.
    ns["xah_client"] = ns["OpenAI"](
        api_key=ns["XAH_API_KEY"],
        base_url=ns["XAH_BASE_URL"],
        timeout=90.0,
        max_retries=0,
    )
    ns["LAB_MAX_ARTICLES"] = 5000
    ns["LAB_MAX_CHUNKS"] = 30000
    download_rows(ns)
    raw = ns["load_news"](SOURCE)
    news = ns["standardize_news"](raw)
    chunks = ns["build_chunks"](news)
    extraction_source = chunks.head(int(ns.get("EXTRACTION_MAX_CHUNKS", 400))).copy()
    print(json.dumps({"rows": len(news), "chunks": len(chunks), "extraction_chunks": len(extraction_source)}), flush=True)

    ns["connect_neo4j"]()
    ns["setup_graph_schema"]()
    print("Starting LLM extraction", flush=True)
    raw_triples, extraction_errors = extract_parallel(ns, extraction_source)
    print(f"Extraction finished: triples={len(raw_triples)} errors={len(extraction_errors)}", flush=True)
    if raw_triples.empty:
        raise RuntimeError("LLM không tạo được relation nào.")
    print("Starting entity resolution", flush=True)
    entity_map, audit = ns["build_resolution_map"](raw_triples)
    triples = ns["canonicalize_triples"](raw_triples, entity_map)
    nodes = ns["build_nodes"](triples)
    print(f"Inserting graph: nodes={len(nodes)} edges={len(triples)}", flush=True)
    ns["bulk_insert_nodes"](nodes)
    ns["bulk_insert_edges"](triples)
    graph_counts, _ = ns["graph_checks"]()

    raw_triples.to_csv(OUT / "raw_triples_5000.csv", index=False)
    triples.to_csv(OUT / "triples_5000.csv", index=False)
    audit.to_csv(OUT / "entity_resolution_audit_5000.csv", index=False)
    extraction_errors.to_csv(OUT / "extraction_errors_5000.csv", index=False)

    print("Starting 50-question evaluation", flush=True)
    result, summary = evaluate(ns, chunks)
    score_cols = ["flat_comprehensiveness", "flat_faithfulness", "flat_multi_hop_reasoning"]
    graph_cols = ["graph_comprehensiveness", "graph_faithfulness", "graph_multi_hop_reasoning"]
    print(json.dumps({
        "graph": graph_counts,
        "rows_evaluated": len(result),
        "flat_mean_score": float(result[score_cols].mean().mean()),
        "graph_mean_score": float(result[graph_cols].mean().mean()),
        "raw_triples": len(raw_triples),
        "canonical_triples": len(triples),
        "extraction_errors": len(extraction_errors),
    }, indent=2), flush=True)


def evaluate_only() -> None:
    """Resume the expensive benchmark after graph preparation has completed."""
    ns = load_notebook_namespace()
    ns["xah_client"] = ns["OpenAI"](
        api_key=ns["XAH_API_KEY"],
        base_url=ns["XAH_BASE_URL"],
        timeout=90.0,
        max_retries=0,
    )
    ns["LAB_MAX_ARTICLES"] = 5000
    ns["LAB_MAX_CHUNKS"] = 30000
    raw = ns["load_news"](SOURCE)
    news = ns["standardize_news"](raw)
    chunks = ns["build_chunks"](news)
    print(json.dumps({"rows": len(news), "chunks": len(chunks)}), flush=True)
    ns["connect_neo4j"]()
    ns["setup_graph_schema"]()
    print("Resuming 50-question evaluation", flush=True)
    result, _ = evaluate(ns, chunks)
    flat_cols = ["flat_comprehensiveness", "flat_faithfulness", "flat_multi_hop_reasoning"]
    graph_cols = ["graph_comprehensiveness", "graph_faithfulness", "graph_multi_hop_reasoning"]
    print(json.dumps({
        "rows_evaluated": len(result),
        "flat_mean_score": float(result[flat_cols].mean().mean()),
        "graph_mean_score": float(result[graph_cols].mean().mean()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    evaluate_only() if "--eval-only" in sys.argv else main()
