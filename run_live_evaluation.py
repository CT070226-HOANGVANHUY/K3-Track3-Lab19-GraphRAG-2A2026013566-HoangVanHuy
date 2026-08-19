"""Run the real Xah generation + LLM-as-a-Judge evaluation.

The notebook contains the pipeline functions; this entry point loads those
definitions, uses the verified 1,000-row live export, and evaluates all 50
Golden questions with the configured OpenAI-compatible provider.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
import json

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Day19_GraphRAG_vs_FlatRAG_Production_Lab_Guide.ipynb"
LIVE_DATA = ROOT / "data" / "hf_live_1000.csv"
GOLDEN_DATA = ROOT / "data" / "golden_dataset.csv"


def load_notebook_namespace() -> dict:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = {"__name__": "live_evaluation", "__file__": str(NOTEBOOK)}
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if "#@title 1.3" in source or "#@title 6.1" in source:
            continue
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def judge_with_retry(ns, question, reference, answer, context):
    last = None
    for attempt in range(3):
        try:
            return ns["judge_answer"](question, reference, answer, context)
        except Exception as exc:
            last = exc
            if attempt < 2:
                sleep(2**attempt)
    raise last


def main() -> None:
    if not LIVE_DATA.exists():
        raise FileNotFoundError(LIVE_DATA)
    ns = load_notebook_namespace()
    ns["DATA_PATH"] = str(LIVE_DATA)
    raw = ns["load_news"](LIVE_DATA)
    news = ns["standardize_news"](raw)
    chunks = ns["build_chunks"](news)

    ns["connect_neo4j"]()
    ns["setup_graph_schema"]()
    nodes = pd.DataFrame(ns["run_cypher"](
        "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, "
        "n.name_norm AS name_norm, n.entity_type AS type, "
        "coalesce(n.aliases, []) AS aliases, "
        "coalesce(n.aliases_norm, []) AS aliases_norm"
    ))
    if nodes.empty:
        raise RuntimeError("Neo4j has no Entity nodes.")
    ns["build_flat_index"](chunks)
    ns["build_entity_matcher"](nodes)

    golden = pd.read_csv(GOLDEN_DATA).fillna("")
    if len(golden) != 50 or golden.reference_answer.str.strip().eq("").any():
        raise RuntimeError("Golden dataset must contain 50 non-empty reference answers.")

    def retrieve(row):
        flat_context, _ = ns["retrieve_flat_context"](row.question, k=6)
        graph_debug = ns["retrieve_graph_context"](
            row.question, max_hops=2, edge_limit=50, return_debug=True
        )
        graph_context = (
            f"=== GRAPH ===\n{graph_debug['context']}\n\n"
            f"=== VECTOR ===\n{flat_context}"
        )
        return {"row": row, "flat_context": flat_context,
                "graph_context": graph_context, "graph_debug": graph_debug}

    def generate(item):
        question = item["row"].question
        item["flat"] = ns["generate_answer"](question, item["flat_context"])
        item["graph"] = ns["generate_answer"](question, item["graph_context"])
        return item

    def evaluate(item):
        row = item["row"]
        flat_judge = judge_with_retry(
            ns, row.question, row.reference_answer,
            item["flat"]["answer"], item["flat_context"]
        )
        graph_judge = judge_with_retry(
            ns, row.question, row.reference_answer,
            item["graph"]["answer"], item["graph_context"]
        )
        return {
            "id": row.id, "group": row.group, "question": row.question,
            "reference_answer": row.reference_answer,
            "flat_answer": item["flat"]["answer"],
            "graph_answer": item["graph"]["answer"],
            "flat_comprehensiveness": flat_judge["comprehensiveness"],
            "graph_comprehensiveness": graph_judge["comprehensiveness"],
            "flat_faithfulness": flat_judge["faithfulness"],
            "graph_faithfulness": graph_judge["faithfulness"],
            "flat_multi_hop_reasoning": flat_judge["multi_hop_reasoning"],
            "graph_multi_hop_reasoning": graph_judge["multi_hop_reasoning"],
            "flat_latency_s": item["flat"]["latency_s"],
            "graph_latency_s": item["graph"]["latency_s"],
            "flat_total_tokens": item["flat"].get("total_tokens"),
            "graph_total_tokens": item["graph"].get("total_tokens"),
            "flat_judge_rationale": flat_judge["rationale"],
            "graph_judge_rationale": graph_judge["rationale"],
            "graph_supernode_events": len(
                item["graph_debug"]["diagnostics"].get("supernode_events", [])
            ),
        }

    rows = list(golden.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=4) as pool:
        retrieved = list(pool.map(retrieve, rows))
    with ThreadPoolExecutor(max_workers=4) as pool:
        generated = list(pool.map(generate, retrieved))

    evaluated = []
    checkpoint = ROOT / "outputs" / "graphrag_eval_actual_checkpoint.csv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate, item) for item in generated]
        for future in as_completed(futures):
            evaluated.append(future.result())
            pd.DataFrame(evaluated).sort_values("id").to_csv(checkpoint, index=False)

    result = pd.DataFrame(evaluated).sort_values("id").reset_index(drop=True)
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
            summary.append({
                "Loại câu hỏi": group, "Metric": metric,
                "Flat RAG": round(flat, 3), "GraphRAG": round(graph, 3),
                "Delta (Graph - Flat)": round(graph - flat, 3),
                "Nhận xét phân tích": (
                    "Flat RAG nhẹ hơn về runtime/token."
                    if metric in {"Latency (s)", "Token usage"} and flat < graph
                    else "GraphRAG cải thiện coverage/chain reasoning."
                    if metric not in {"Latency (s)", "Token usage"} and graph > flat
                    else "Hai phương pháp gần nhau."
                ),
            })

    result.to_csv(ROOT / "outputs" / "graphrag_eval_results.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "outputs" / "graphrag_vs_flatrag_summary.csv", index=False)
    print(json.dumps({
        "rows": len(result),
        "mean_flat_score": float(result[[
            "flat_comprehensiveness", "flat_faithfulness", "flat_multi_hop_reasoning"
        ]].mean().mean()),
        "mean_graph_score": float(result[[
            "graph_comprehensiveness", "graph_faithfulness", "graph_multi_hop_reasoning"
        ]].mean().mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
