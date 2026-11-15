"""Unified Command-Line Interface (CLI) for the Toolbench suite.

Orchestrates the entire toolbench experimental workflow:
- `list-tools`: Discover, namespace, and display tools from configured MCP servers.
- `generate`: Synthesize candidate prompts with 3-tier anti-leakage filtering.
- `pilot`: Execute pre-review quality gate check on raw candidate prompts.
- `review`: Interactive manual curation and verification of prompts.
- `split`: Complete-linkage clustering and Hamilton-style dev/held-out dataset split.
- `select-k`: Sweep candidate k values on dev set to calibrate retrieval recall.
- `rewrite-descriptions`: LLM-assisted tool description rewriting with word limit truncation.
- `run`: Execute benchmark evaluation (Setups A, B, C, D) with schema retry.
- `compare`: 2x2 factorial statistical comparison with McNemar, bootstrap CIs, and Holm correction.
- `analyze-failures`: Categorize and inspect failure modes for an evaluation run.
- `validate`: Validate benchmark result or comparison report files against formal schemas.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from toolbench.config import ToolbenchConfig, load_config
from toolbench.dataset import review_prompt_batch, save_dataset_split, split_dataset
from toolbench.embeddings import get_default_embedding_helper
from toolbench.generate import generate_prompts
from toolbench.mcp_client import MultiServerAggregator, ToolInfo
from toolbench.output import (
    console,
    print_error,
    print_info,
    print_prompt_panel,
    print_success,
    print_warning,
    render_tools_table,
)
from toolbench.pilot import run_pilot_gate
from toolbench.providers import get_provider
from toolbench.report import (
    analyze_failures,
    append_heldout_audit_log,
    compare_setups,
    render_comparison_tables,
)
from toolbench.results_schema import validate_results_file
from toolbench.retrieve import (
    ToolRetriever,
    calibrate_k,
    compare_retrieval_methods,
    render_calibration_table,
)
from toolbench.rewrite import load_rewritten_descriptions, rewrite_descriptions
from toolbench.run import run_benchmark

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="toolbench",
    help="Scientific MCP tool selection benchmarking and evaluation CLI.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_tools_from_servers(config: ToolbenchConfig) -> list[ToolInfo]:
    """Connect to configured MCP servers and discover tools."""
    aggregator = MultiServerAggregator(config.mcp)
    return asyncio.run(aggregator.aggregate_async())


@app.command("list-tools")
def list_tools_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="File to write tools JSON"),
    ] = "data/tools.json",
) -> None:
    """Connect to configured MCP servers, discover tools, save data/tools.json, and show summary."""
    cfg = load_config(config_path)
    print_info(f"Connecting to {len(cfg.mcp)} MCP server(s)...")

    try:
        tools = _load_tools_from_servers(cfg)
    except Exception as err:
        print_error(f"Failed to discover tools: {err}")
        raise typer.Exit(code=1) from err

    # Save to data/tools.json
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps([t.model_dump() for t in tools], indent=2), encoding="utf-8")

    # Server counts breakdown
    server_counts: dict[str, int] = {}
    for t in tools:
        server = t.server_name or "unknown"
        server_counts[server] = server_counts.get(server, 0) + 1

    table = render_tools_table(tools, title=f"Discovered Tools ({len(tools)} total)")
    console.print(table)

    server_summary = ", ".join(f"{s}: {c}" for s, c in sorted(server_counts.items()))
    print_info(f"Tools per server: {server_summary}")
    print_success(f"Saved {len(tools)} tools to '{out_p}'.")

    # Warn if total under 40
    if len(tools) < 40:
        print_warning(
            f"Total discovered tools ({len(tools)}) is under 40. "
            "Retrieval may experience a ceiling effect; consider adding another server."
        )


@app.command("generate")
def generate_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="Destination path for raw prompts JSON"),
    ] = "data/raw_prompts.json",
    use_mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock provider instead of Gemini"),
    ] = False,
    incremental: Annotated[
        bool,
        typer.Option("--incremental", help="Append new prompts to existing dataset"),
    ] = False,
) -> None:
    """Generate synthetic candidate prompts with anti-leakage validation (F2)."""
    cfg = load_config(config_path)
    if use_mock:
        cfg.model.provider = "mock"
    provider = get_provider(cfg)
    tools = _load_tools_from_servers(cfg)
    embed_helper = get_default_embedding_helper()

    records = generate_prompts(
        tools=tools,
        provider=provider,
        embedding_helper=embed_helper,
        config=cfg.generation,
    )

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    final_records = [r.model_dump() for r in records]
    if incremental and out_p.exists():
        try:
            existing = json.loads(out_p.read_text(encoding="utf-8"))
            existing_texts = {p.get("prompt_text") for p in existing}
            added = [r for r in final_records if r.get("prompt_text") not in existing_texts]
            final_records = existing + added
            print_info(
                f"Incremental mode: appended {len(added)} new prompts to {len(existing)} existing."
            )
        except Exception as err:
            logger.warning("Could not read existing prompts for incremental merge: %s", err)

    out_p.write_text(json.dumps(final_records, indent=2), encoding="utf-8")
    print_success(
        f"Generated raw candidate prompts saved ({len(final_records)} total) to '{out_p}'."
    )


@app.command("pilot")
def pilot_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    data_file: Annotated[
        str,
        typer.Option("--data", "-d", help="Path to raw candidate prompts JSON"),
    ] = "data/raw_prompts.json",
    use_mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock provider"),
    ] = False,
) -> None:
    """Execute pre-review Pilot Gate quality check on raw prompts."""
    cfg = load_config(config_path)
    if use_mock:
        cfg.model.provider = "mock"
    p_path = Path(data_file)
    if not p_path.exists():
        print_error(f"Raw prompts file not found at '{p_path}'. Run 'toolbench generate' first.")
        raise typer.Exit(code=1)

    raw_prompts = json.loads(p_path.read_text(encoding="utf-8"))
    tools = _load_tools_from_servers(cfg)
    provider = get_provider(cfg)

    passed, summary = run_pilot_gate(
        tools=tools,
        provider=provider,
        config=cfg,
        prompts=raw_prompts,
    )
    if not passed:
        print_error("Pilot Gate failed: ceiling effect detected. Expand tool set before review.")
        raise typer.Exit(code=1)
    print_success(
        f"Pilot Gate passed! Tool accuracy {summary.tool_accuracy_pct:.1f}% "
        "is in viable benchmark range."
    )


@app.command("review")
def review_command(
    input_file: Annotated[
        str,
        typer.Option("--input", "-i", help="Raw prompts JSON file to review"),
    ] = "data/raw_prompts.json",
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="Destination path for verified prompts"),
    ] = "data/verified_prompts.json",
) -> None:
    """Interactively review and curate candidate prompts into verified dataset (F3)."""
    in_p = Path(input_file)
    if not in_p.exists():
        print_error(f"Input file not found at '{in_p}'.")
        raise typer.Exit(code=1)

    raw_prompts = json.loads(in_p.read_text(encoding="utf-8"))
    print_info(f"Loaded {len(raw_prompts)} prompts for interactive review.")

    def cli_decision(p: dict, idx: int, total: int) -> tuple[str, dict | None]:
        print_prompt_panel(
            prompt_id=str(p.get("prompt_id", idx + 1)),
            prompt_text=p.get("prompt_text", p.get("prompt", "")),
            expected_tool=p.get("expected_tool", ""),
            expected_args=p.get("expected_args", {}),
            description=f"Reviewing prompt {idx + 1} of {total}",
        )
        choice = typer.prompt("Decision ([a]pprove, [e]dit args, [s]kip)", default="a").strip()
        if choice.lower().startswith("e"):
            default_json = json.dumps(p.get("expected_args", {}))
            raw_args = typer.prompt("Enter new JSON args", default=default_json)
            try:
                new_args = json.loads(raw_args)
                return "e", new_args
            except Exception:
                print_warning("Invalid JSON. Approving original args.")
                return "a", None
        return choice, None

    verified = review_prompt_batch(
        raw_prompts=raw_prompts,
        decision_callback=cli_decision,
        output_path=output_file,
    )
    print_success(
        f"Review session complete: {len(verified)} verified prompts saved to '{output_file}'."
    )


@app.command("split")
def split_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    verified_file: Annotated[
        str,
        typer.Option("--input", "-i", help="Verified prompts JSON file"),
    ] = "data/verified_prompts.json",
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", "-d", help="Directory to save split datasets and hash"),
    ] = "data",
) -> None:
    """Cluster near-duplicates and split verified prompts into dev and held-out sets (F3)."""
    cfg = load_config(config_path)
    in_p = Path(verified_file)
    if not in_p.exists():
        print_error(f"Verified prompts file not found at '{in_p}'. Run 'toolbench review' first.")
        raise typer.Exit(code=1)

    verified = json.loads(in_p.read_text(encoding="utf-8"))
    dev, heldout, h_hash = split_dataset(verified, config=cfg.split)
    save_dataset_split(dev, heldout, h_hash, data_dir=data_dir)
    print_success(
        f"Dataset split saved: dev={len(dev)}, heldout={len(heldout)}, SHA-256={h_hash[:12]}..."
    )


@app.command("select-k")
def select_k_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", "-d", help="Directory containing dev_prompts.json"),
    ] = "data",
) -> None:
    """Sweep candidate k values on dev set to calibrate retrieval recall (F5)."""
    cfg = load_config(config_path)
    dev_p = Path(data_dir) / "dev_prompts.json"
    if not dev_p.exists():
        print_error(f"Dev dataset not found at '{dev_p}'. Run 'toolbench split' first.")
        raise typer.Exit(code=1)

    dev_prompts = json.loads(dev_p.read_text(encoding="utf-8"))
    tools = _load_tools_from_servers(cfg)
    embed_helper = get_default_embedding_helper()

    selected_k, recall_curve = calibrate_k(
        prompts=dev_prompts,
        tools=tools,
        candidate_k_values=cfg.retrieval.candidate_k,
        target_recall=cfg.retrieval.target_recall,
        embedding_helper=embed_helper,
        show_table=False,
    )

    table = render_calibration_table(recall_curve, selected_k, cfg.retrieval.target_recall)
    console.print(table)

    # Save k_selection.json
    k_sel_path = Path(data_dir) / "k_selection.json"
    k_sel_data = {
        "selected_k": selected_k,
        "target_recall": cfg.retrieval.target_recall,
        "recall_curve": {str(k): rec for k, rec in sorted(recall_curve.items())},
        "total_tools": len(tools),
        "dev_prompts_count": len(dev_prompts),
    }
    k_sel_path.write_text(json.dumps(k_sel_data, indent=2), encoding="utf-8")
    print_success(
        f"Calibrated optimal retrieval parameter: k={selected_k} (saved to '{k_sel_path}')"
    )


@app.command("compare-retrieval")
def compare_retrieval_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", "-d", help="Directory containing dev_prompts.json"),
    ] = "data",
    methods: Annotated[
        str,
        typer.Option(
            "--methods",
            "-m",
            help="Comma-separated retrieval methods (dense,bm25,hybrid,cross_encoder)",
        ),
    ] = "dense,bm25,hybrid",
    k_values: Annotated[
        str | None,
        typer.Option("--k", help="Comma-separated candidate k values (e.g. '5,10,15,20')"),
    ] = None,
) -> None:
    """Compare Recall@k across multiple retrieval paradigms on dev dataset (D1)."""
    cfg = load_config(config_path)
    dev_p = Path(data_dir) / "dev_prompts.json"
    if not dev_p.exists():
        print_error(f"Dev dataset not found at '{dev_p}'. Run 'toolbench split' first.")
        raise typer.Exit(code=1)

    dev_prompts = json.loads(dev_p.read_text(encoding="utf-8"))
    tools = _load_tools_from_servers(cfg)
    embed_helper = get_default_embedding_helper()

    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    if k_values:
        candidates = [int(k.strip()) for k in k_values.split(",") if k.strip().isdigit()]
    else:
        candidates = cfg.retrieval.candidate_k

    compare_retrieval_methods(
        prompts=dev_prompts,
        tools=tools,
        candidate_k_values=candidates,
        methods=method_list,
        embedding_helper=embed_helper,
        show_table=True,
    )


@app.command("rewrite-descriptions")
def rewrite_descriptions_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    output_file: Annotated[
        str,
        typer.Option("--output", "-o", help="File to store frozen rewritten descriptions"),
    ] = "data/rewritten_descriptions.json",
    use_mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock provider"),
    ] = False,
) -> None:
    """Generate and freeze distinctive rewritten tool descriptions with word limit (F6)."""
    cfg = load_config(config_path)
    if use_mock:
        cfg.model.provider = "mock"
    tools = _load_tools_from_servers(cfg)
    provider = get_provider(cfg)

    rewritten_map = rewrite_descriptions(
        tools=tools,
        provider=provider,
        config=cfg.generation,
        output_path=output_file,
        max_words=200,
    )
    print_success(f"Rewrote descriptions for {len(rewritten_map)} tools frozen to '{output_file}'.")


@app.command("run")
def run_command(
    setup: Annotated[
        str,
        typer.Option("--setup", "-s", help="Factorial setup letter: A, B, C, or D"),
    ] = "A",
    split: Annotated[
        str,
        typer.Option("--split", help="Target dataset split: dev, heldout, or pilot"),
    ] = "heldout",
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", "-d", help="Directory containing dataset files"),
    ] = "data",
    results_dir: Annotated[
        str,
        typer.Option("--results-dir", "-r", help="Directory to save evaluation results"),
    ] = "results",
    use_mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock provider"),
    ] = False,
    acknowledge_ceiling: Annotated[
        bool,
        typer.Option("--acknowledge-ceiling", help="Bypass dev-set accuracy ceiling check"),
    ] = False,
    allow_unverified: Annotated[
        bool,
        typer.Option("--allow-unverified", help="Permit scoring unverified prompts"),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Bypass response cache for generation and evaluation"),
    ] = False,
    clear_cache: Annotated[
        bool,
        typer.Option("--clear-cache", help="Clear response cache before running"),
    ] = False,
) -> None:
    """Execute benchmark evaluation across prompts for a given setup and split (F4+F7)."""
    setup_clean = setup.upper().strip()
    if setup_clean not in ["A", "B", "C", "D"]:
        print_error(f"Invalid setup '{setup}'. Choose from A, B, C, or D.")
        raise typer.Exit(code=1)

    cfg = load_config(config_path)
    if use_mock:
        cfg.model.provider = "mock"
    tools = _load_tools_from_servers(cfg)
    provider = get_provider(cfg)

    rewritten_map: dict[str, str] = {}
    if setup_clean in ["B", "D"]:
        rewritten_map = load_rewritten_descriptions(Path(data_dir) / "rewritten_descriptions.json")

    # Prepare tools with rewritten descriptions for B and D
    tools_for_run = tools
    if setup_clean in ["B", "D"]:
        tools_for_run = [
            ToolInfo(
                name=t.name,
                description=rewritten_map.get(t.name, t.description),
                input_schema=t.input_schema,
            )
            for t in tools
        ]

    tool_selector_fn = None
    if setup_clean in ["C", "D"]:
        embed_helper = get_default_embedding_helper()
        retriever = ToolRetriever(
            tools=tools_for_run,
            embedding_helper=embed_helper,
            descriptions=rewritten_map if setup_clean == "D" else None,
        )
        k_val = cfg.retrieval.candidate_k[0] if cfg.retrieval.candidate_k else 5

        def _select(q: str, t_list: list[ToolInfo]) -> list[ToolInfo]:
            return retriever.retrieve_top_k(q, k=k_val)

        tool_selector_fn = _select

    run_res = run_benchmark(
        tools=tools_for_run,
        provider=provider,
        config=cfg,
        split=split,
        setup=setup_clean,
        data_dir=data_dir,
        results_dir=results_dir,
        acknowledge_ceiling=acknowledge_ceiling,
        allow_unverified=allow_unverified,
        no_cache=no_cache,
        clear_cache=clear_cache,
        tool_selector=tool_selector_fn,
    )

    if split == "heldout" and run_res.heldout_hash:
        append_heldout_audit_log(
            {
                "run_id": run_res.run_id,
                "setup": setup_clean,
                "model": provider.model_name,
                "heldout_hash": run_res.heldout_hash,
                "prompt_count": run_res.total_prompts,
                "tool_accuracy": run_res.tool_accuracy,
                "args_strict_match_rate": run_res.strict_match_rate,
                "args_valid_post_retry_rate": run_res.post_retry_validity_rate,
            },
            log_file=Path(results_dir) / "heldout_log.jsonl",
        )


@app.command("compare")
def compare_command(
    results_dir: Annotated[
        str,
        typer.Option("--results-dir", "-r", help="Directory containing result JSON files"),
    ] = "results",
) -> None:
    """Compare evaluation outcomes across Setups A, B, C, D with statistical tests (F8)."""
    r_dir = Path(results_dir)
    if not r_dir.exists():
        print_error(f"Results directory '{r_dir}' not found.")
        raise typer.Exit(code=1)

    # Automatically identify most recent results for each setup
    results_map: dict[str, Path] = {}
    for letter in ["A", "B", "C", "D"]:
        matches = sorted(r_dir.glob(f"{letter}_heldout_*.json"))
        if matches:
            results_map[letter] = matches[-1]

    if len(results_map) < 2:
        print_error(
            f"Need at least 2 setup result files for comparison, found: {list(results_map.keys())}"
        )
        raise typer.Exit(code=1)

    report = compare_setups(results_map, output_dir=results_dir)
    render_comparison_tables(report)


@app.command("analyze-failures")
def analyze_failures_command(
    result_file: Annotated[
        str,
        typer.Argument(help="Path to evaluation result JSON file (e.g. results/A_heldout_*.json)"),
    ],
) -> None:
    """Inspect and categorize failure modes for an evaluation run (F9)."""
    p = Path(result_file)
    if not p.exists():
        print_error(f"Result file '{p}' not found.")
        raise typer.Exit(code=1)

    data = json.loads(p.read_text(encoding="utf-8"))
    analysis = analyze_failures(data)

    console.print(f"\n[bold cyan]Failure Mode Analysis for Setup {analysis['setup']}[/]")
    console.print(f"Total Prompts Evaluated: {analysis['total_prompts']}\n")

    for cat, count in analysis["category_counts"].items():
        pct = analysis["category_percentages"].get(cat, 0.0)
        style = "green" if cat == "pass" else "yellow"
        console.print(f"[{style}]{cat:30s}[/]: {count:4d} ({pct:5.1f}%)")

    # Build and write failures.json alongside the result file
    failures_out = p.parent / f"failures_{analysis['setup']}.json"
    failures_out.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print_success(f"Saved failure analysis to '{failures_out}'.")


@app.command("validate")
def validate_command(
    file_path: Annotated[
        str,
        typer.Argument(help="Path to benchmark result or comparison report JSON file"),
    ],
) -> None:
    """Validate a results file against formal Pydantic schema contracts (F11f)."""
    p = Path(file_path)
    try:
        kind, model = validate_results_file(p)
        print_success(f"File '{p.name}' is valid schema contract for kind: '{kind}'.")
    except Exception as err:
        print_error(f"Validation failed for '{p}': {err}")
        raise typer.Exit(code=1) from err


@app.command("proxy")
def proxy_command(
    config_path: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to config.yaml"),
    ] = None,
) -> None:
    """Launch the standalone Retrieval Proxy MCP Server over stdio (D3)."""
    from toolbench.proxy import create_proxy_server

    cfg = load_config(config_path)
    tools = _load_tools_from_servers(cfg)
    embed_helper = get_default_embedding_helper()
    retriever = ToolRetriever(tools=tools, embedding_helper=embed_helper)

    server = create_proxy_server(tools=tools, retriever=retriever)
    print_info(f"Starting Toolbench Retrieval Proxy for {len(tools)} tools on stdio...")
    asyncio.run(server.run_stdio_async())


@app.command("tui")
def tui_command(
    results_dir: Annotated[
        str,
        typer.Option(
            "--results-dir",
            "-d",
            help="Directory containing benchmark result JSON files",
        ),
    ] = "results",
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Render full dashboard once without interactive prompt",
        ),
    ] = False,
) -> None:
    """Launch the interactive Terminal UI (TUI) Dashboard for results exploration (D4)."""
    from toolbench.tui import TerminalDashboard, run_interactive_dashboard

    dashboard = TerminalDashboard(results_dir=results_dir)
    if non_interactive:
        console.print(dashboard.render_full_dashboard())
    else:
        run_interactive_dashboard(dashboard)


@app.command("dashboard")
def dashboard_command(
    results_dir: Annotated[
        str,
        typer.Option("--results-dir", "-r", help="Directory containing result JSON files"),
    ] = "results",
    dashboard_data_dir: Annotated[
        str,
        typer.Option("--dashboard-dir", "-d", help="Dashboard data directory"),
    ] = "dashboard/src",
) -> None:
    """Validate benchmark results and copy them into the dashboard data directory (F10)."""
    r_dir = Path(results_dir)
    d_dir = Path(dashboard_data_dir)
    d_dir.mkdir(parents=True, exist_ok=True)

    results_map: dict[str, Any] = {}
    for letter in ("A", "B", "C", "D"):
        candidates = sorted(r_dir.glob(f"{letter}_*.json"))
        if candidates:
            target = candidates[-1]
            try:
                validate_results_file(target)
                data = json.loads(target.read_text(encoding="utf-8"))
                results_map[letter] = data
            except Exception as err:
                print_warning(f"Could not validate {target.name}: {err}")

    if not results_map:
        print_error("No valid benchmark results found in results directory to sync.")
        raise typer.Exit(code=1)

    out_json = d_dir / "results.json"
    out_json.write_text(json.dumps(results_map, indent=2), encoding="utf-8")
    print_success(
        f"Validated and synced results for setups {list(results_map.keys())} to '{out_json}'."
    )
    print_info("To view the web dashboard, run: cd dashboard && npm run dev (or npm run preview)")


@app.command("clear-cache")
def clear_cache_command(
    all_caches: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="Clear both LLM response cache and embedding vector cache",
        ),
    ] = False,
) -> None:
    """Remove response (and optionally embedding) cache (F11a)."""
    import shutil

    from toolbench.cache import DiskResponseCache

    DiskResponseCache().clear()
    print_success("Cleared LLM response disk cache.")

    if all_caches:
        emb_cache = Path(".cache/toolbench/embeddings")
        if emb_cache.exists():
            shutil.rmtree(emb_cache)
            print_success("Cleared embedding vector cache.")


if __name__ == "__main__":
    app()
