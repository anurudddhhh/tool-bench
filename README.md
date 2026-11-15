# toolbench

A reproducible, scientific benchmarking suite to measure and improve LLM tool selection in Model Context Protocol (MCP) ecosystems using a $2 \times 2$ factorial experimental design.

---

## 1. Experimental Design ($2 \times 2$ Factorial)

Toolbench isolates and measures the individual and interaction effects of **embedding-based retrieval** and **LLM description rewriting**:

| Setup | Description Source | Tool Exposure Strategy | Experimental Purpose |
| :---: | :--- | :--- | :--- |
| **A** | Original MCP descriptions | Full exposure (all discovered tools) | **Baseline**: Standard MCP client behavior |
| **B** | LLM-rewritten descriptions | Full exposure (all discovered tools) | **Isolates Rewriting**: Effect of distinctive descriptions ($\le 200$ words) |
| **C** | Original MCP descriptions | Top-$k$ retrieved tools | **Isolates Retrieval**: Effect of cosine similarity retrieval |
| **D** | LLM-rewritten descriptions | Top-$k$ retrieved tools | **Combined Treatment**: Retrieval over rewritten descriptions using calibrated $k$ from C |

### Methodological Invariants
- **Free Tier Only**: Zero paid API calls. Uses Gemini `gemini-3.5-flash-lite` (15 RPM, 500 RPD, 250K TPM) resets midnight Pacific Time (`America/Los_Angeles`).
- **Identical Evaluator**: The same Gemini model and version is used across all setups A, B, C, and D.
- **Local Embeddings**: Local CPU `sentence-transformers` (`all-MiniLM-L6-v2`). Zero external embedding API calls.
- **Strict Temperature Control**: Evaluation runs strictly at temperature `0.0`. Prompt generation and rewriting run at temperature `0.7`.
- **Frozen Held-Out Set**: Stratified Hamilton-style dev/held-out split ($\ge 50$ prompts, $\ge 10$ distinct multi-prompt tools, complete-linkage cluster separation at similarity $\ge 0.85$, zero cluster leakage). Frozen with SHA-256 integrity digest.
- **Single-Turn Schema Retry**: If arguments fail JSON Schema validation, a single retry prompt is issued with honest tool schemas. Retries are tracked separately from network backoffs.
- **Paired Statistical Testing**: Primary comparisons ($A \text{ vs } C$, $A \text{ vs } B$, $A \text{ vs } D$) evaluated via paired McNemar's tests (exact binomial for discordant pairs $< 25$, Edwards' continuity-corrected $\chi^2$ for $\ge 25$) with step-down Holm-Bonferroni family-wise error rate control.
- **Empirical Uncertainty**: 95% bootstrap confidence intervals ($2,000$ resamples) on accuracy deltas ($\text{Acc}_2 - \text{Acc}_1$).
- **Append-Only Audit Trail**: Every held-out run records a single-line JSON entry in `results/heldout_log.jsonl`.

---

## 2. Architecture & Subsystems

```
toolbench/
├── src/toolbench/
│   ├── cache.py           # F11a: Atomic SHA-256 disk cache for LLM responses
│   ├── rate_limit.py      # F11b: Sliding-window RPM & Pacific midnight RPD limiter with jitter backoff
│   ├── config.py          # F11c: Pydantic v2 configuration with YAML persistence
│   ├── providers/         # F11d: GeminiProvider (v1beta function calling) & deterministic MockProvider
│   ├── output.py          # F11e: Rich tables, progress indicators, and prompt cards
│   ├── results_schema.py  # F11f: Formal Pydantic schemas for benchmark outputs and reports
│   ├── embeddings.py      # F11h: Local sentence-transformers CPU embedder & atomic .npy vector cache
│   ├── mcp_client.py      # F1, F1.5: Async MCP client, multi-server stdio aggregator, namespacing
│   ├── generate.py        # F2: Synthetic prompt generator with 3-tier anti-leakage filters
│   ├── pilot.py           # Pilot Gate: Pre-review calibration check on 15–20 raw prompts
│   ├── dataset.py         # F3: Complete-linkage clustering, manual review CLI, and Hamilton split
│   ├── run.py             # F4+F7: Benchmark execution engine, schema validation, and retry hook
│   ├── retrieve.py        # F5: Cosine similarity retriever & dev-split k-sweep calibration
│   ├── rewrite.py         # F6: LLM description rewriter with <= 200 words sentence truncation
│   └── report.py          # F8, F9: 2x2 comparison, McNemar, bootstrap CIs, Holm correction, failure analysis
└── tests/                 # 125 automated unit and end-to-end integration tests
```

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.12+
- `uv` package manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
- Node.js (for local MCP servers: `filesystem`, `memory`, `git`, `fetch`, `time`)

### Clone & Setup
```bash
git clone https://github.com/anurudddhhh/tool-bench.git
cd tool-bench

# Install all dependencies into virtual environment
uv sync --all-extras --dev

# Configure API credentials
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### Running Tests
Execute the complete test suite (all 125 tests run 100% offline using `MockProvider`):
```bash
uv run pytest -v
```

Execute static analysis and style checks:
```bash
uv run ruff check .
uv run ruff format --check .
```

---

## 4. Docker Usage

Build and run via Docker:
```bash
# Build the container
docker build -t toolbench .

# Run test suite inside container
docker run --rm toolbench -c "pytest -v"

# Execute CLI help
docker run --rm toolbench --help
```

---

## 5. License

Apache 2.0. See [LICENSE](LICENSE) for details.
