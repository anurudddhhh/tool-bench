"""End-to-end integration test simulating the entire toolbench lifecycle offline (F11g).

Executes a complete dry-run across:
1. Tool aggregation & namespacing.
2. Synthetic prompt generation & 3-tier anti-leakage validation (F2).
3. Complete-linkage clustering and stratified dev/held-out split (F3).
4. Embedding-based k-calibration on dev split (F5).
5. Description rewriting and word-limit truncation (F6).
6. Factorial benchmarking (Setups A, B, C, D) at evaluation temperature 0.0 with
   schema retry (F4+F7).
7. Factorial comparison, McNemar paired tests, Holm correction, and failure breakdown (F8, F9).
8. Formal schema validation on generated artifacts (F11f).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from toolbench.config import GenerationConfig, SplitConfig
from toolbench.dataset import (
    VerifiedPromptRecord,
    save_dataset_split,
    split_dataset,
)
from toolbench.embeddings import EmbeddingHelper
from toolbench.generate import generate_prompts
from toolbench.mcp_client import ToolInfo
from toolbench.providers.mock import MockProvider
from toolbench.report import append_heldout_audit_log, compare_setups
from toolbench.results_schema import validate_results_file
from toolbench.retrieve import ToolRetriever, calibrate_k
from toolbench.rewrite import rewrite_descriptions
from toolbench.run import run_benchmark


class MockDeterministicEmbeddingHelper(EmbeddingHelper):
    """Offline deterministic mock embedding helper returning distinct unit vectors."""

    def __init__(self, cache_dir: Path | str) -> None:
        super().__init__(model_name="mock-embedder", cache_dir=cache_dir)
        self._model = MagicMock()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for t in texts:
            # Deterministic hash to 3D unit vector
            val = abs(hash(t)) % 1000 / 1000.0
            vec = np.array([val, 1.0 - val, 0.5], dtype=np.float32)
            norm = np.linalg.norm(vec)
            vectors.append(vec / (norm if norm > 0 else 1.0))
        return np.array(vectors, dtype=np.float32)


def test_full_toolbench_pipeline_end_to_end(tmp_path: Path):
    # ---------------------------------------------------------
    # 1. MCP Tools Setup
    # ---------------------------------------------------------
    # 10 tools across various domains to satisfy split requirements
    tools = [
        ToolInfo(
            name=f"tool_{i:02d}",
            description=f"Performs automated operation type {i} on system resources.",
            input_schema={
                "type": "object",
                "properties": {"arg_val": {"type": "string"}},
                "required": ["arg_val"] if i % 2 == 0 else [],
            },
        )
        for i in range(10)
    ]

    # ---------------------------------------------------------
    # 2. Synthetic Prompt Generation (F2)
    # ---------------------------------------------------------
    mock_gen_json = json.dumps(
        [
            {
                "prompt": f"Operation task prompt for tool {i}",
                "expected_tool": f"tool_{i:02d}",
                "expected_args": {"arg_val": f"val_{i}"} if i % 2 == 0 else {},
            }
            for i in range(10)
        ]
    )
    gen_provider = MockProvider(simulate_text_only=mock_gen_json)
    embed_helper = MockDeterministicEmbeddingHelper(cache_dir=tmp_path / "embed_cache")

    generated = generate_prompts(
        tools=tools,
        provider=gen_provider,
        embedding_helper=embed_helper,
        config=GenerationConfig(count=40, temperature=0.7),
    )
    assert len(generated) == 40

    # ---------------------------------------------------------
    # 3. Dataset Review, Clustering & Split (F3)
    # ---------------------------------------------------------
    verified_prompts: list[VerifiedPromptRecord] = []
    # 100 prompts (10 per tool) ensuring each tool has multiple prompts
    for i in range(100):
        target_tool = tools[i % len(tools)]
        has_arg = "arg_val" in target_tool.input_schema["properties"]
        args = {"arg_val": f"param_{i}"} if has_arg else {}
        verified_prompts.append(
            VerifiedPromptRecord(
                prompt_id=i + 1,
                prompt_text=f"Execute action number {i} for {target_tool.name}",
                expected_tool=target_tool.name,
                expected_args=args,
            )
        )

    prompt_dicts = [p.model_dump() for p in verified_prompts]
    dev_prompts, heldout_prompts, heldout_hash = split_dataset(
        prompt_dicts,
        config=SplitConfig(
            heldout_size=50,
            min_dev_size=40,
            seed=42,
        ),
    )

    data_dir = tmp_path / "data"
    save_dataset_split(dev_prompts, heldout_prompts, heldout_hash, data_dir=data_dir)
    assert heldout_hash is not None
    assert len(heldout_prompts) >= 50
    assert len(dev_prompts) >= 40

    # ---------------------------------------------------------
    # 4. Calibration of k on Dev Split (F5)
    # ---------------------------------------------------------
    k_val, recall_curve = calibrate_k(
        prompts=dev_prompts,
        tools=tools,
        embedding_helper=embed_helper,
        candidate_k_values=[2, 5, 8],
        target_recall=0.50,
        show_table=False,
    )
    assert k_val in [2, 5, 8]
    assert len(recall_curve) == 3

    # ---------------------------------------------------------
    # 5. Description Rewriting (F6)
    # ---------------------------------------------------------
    rewriter_provider = MockProvider(
        simulate_text_only="Distinct rewritten description detailing precise trigger conditions."
    )
    rewritten_path = tmp_path / "data" / "rewritten_descriptions.json"
    rewritten_map = rewrite_descriptions(
        tools=tools,
        provider=rewriter_provider,
        output_path=rewritten_path,
        max_words=200,
    )
    assert len(rewritten_map) == len(tools)
    assert rewritten_path.exists()

    # ---------------------------------------------------------
    # 6. Benchmark Execution (Setups A, B, C, D) (F4+F7)
    # ---------------------------------------------------------
    # Provider mapping keyword to tool call
    eval_provider = MockProvider(
        tool_mapping={
            "tool_00": ("tool_00", {"arg_val": "param_0"}),
            "tool_01": ("tool_01", {}),
        },
        default_tool="tool_00",
        default_args={"arg_val": "default_val"},
    )

    results_dir = tmp_path / "results"
    setups = ["A", "B", "C", "D"]
    result_files: dict[str, Path] = {}

    for s in setups:
        tools_for_run = tools
        if s in ["B", "D"]:
            tools_for_run = [
                ToolInfo(
                    name=t.name,
                    description=rewritten_map.get(t.name, t.description),
                    input_schema=t.input_schema,
                )
                for t in tools
            ]

        tool_selector_fn = None
        if s in ["C", "D"]:
            retriever = ToolRetriever(
                tools=tools_for_run,
                embedding_helper=embed_helper,
                descriptions=rewritten_map if s == "D" else None,
            )

            def make_selector(r: ToolRetriever, k: int):
                def _select(q: str, t_list: list[ToolInfo]) -> list[ToolInfo]:
                    return r.retrieve_top_k(q, k=k)

                return _select

            tool_selector_fn = make_selector(retriever, k_val)

        run_res = run_benchmark(
            tools=tools_for_run,
            provider=eval_provider,
            split="heldout",
            setup=s,
            data_dir=data_dir,
            results_dir=results_dir,
            tool_selector=tool_selector_fn,
        )

        assert run_res.output_file is not None
        res_path = Path(run_res.output_file)
        assert res_path.exists()
        result_files[s] = res_path

        # Verify schema validity of benchmark result file (F11f)
        kind, model = validate_results_file(res_path)
        assert kind == "benchmark"
        assert model.setup == s

        # Append to audit log
        append_heldout_audit_log(
            {
                "run_id": run_res.run_id,
                "setup": s,
                "model": "mock-model",
                "heldout_hash": heldout_hash,
                "prompt_count": len(heldout_prompts),
                "tool_accuracy": run_res.tool_accuracy,
                "args_strict_match_rate": run_res.strict_match_rate,
                "args_valid_post_retry_rate": run_res.post_retry_validity_rate,
            },
            log_file=results_dir / "heldout_log.jsonl",
        )

    # ---------------------------------------------------------
    # 7. Comparison Report & Failure Analysis (F8, F9)
    # ---------------------------------------------------------
    report = compare_setups(result_files, output_dir=results_dir)

    assert report["heldout_hash"] == heldout_hash
    assert len(report["primary_comparisons"]) == 3
    assert len(report["exploratory_comparisons"]) == 3
    assert len(report["failure_analyses"]) == 4

    # Verify audit log was created
    audit_log = results_dir / "heldout_log.jsonl"
    assert audit_log.exists()
    lines = audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    # Find and validate comparison JSON file (F11f)
    comp_files = list(results_dir.glob("comparison_*.json"))
    assert len(comp_files) == 1
    c_kind, c_model = validate_results_file(comp_files[0])
    assert c_kind == "comparison"
    assert c_model.heldout_hash == heldout_hash
