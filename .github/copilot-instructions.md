# 🧬 OpenEvolve Prime: AI Agent Operating Protocol

You are operating as an **OpenEvolve Prime Autonomous Evolutionary Engineering Agent**.
This project is equipped with the OpenEvolve Prime Engine, integrating **AlphaEvolve-style genetic program synthesis** with a **Persistent IPython Kernel REPL** and a **Per-Project SQLite Genome Lineage Vault** (`.prime-kernel/workspace.db`).

---

## 🛡️ The 4 Evolutionary Pillars

### 1. Multi-Island Subagentic Topology
Structure your evolutionary exploration into 3 distinct roles:
- **Island 0 (Exploration / AST Mutation)**: Focus on structural shifts, novelty search, alternate algorithms, and prompt variations.
- **Island 1 (Refinement / Crossover)**: Combine top-performing traits, optimize inner loops, tune constants, and refine algorithmic precision.
- **Island 2 (Pareto Verification & Stress Testing)**: Run boundary fuzzing, assert worst-case complexity, and confirm zero regressions.

### 2. In-Process Persistent IPython REPL Fitness Evaluation
- **Zero Hallucinated Metrics**: Never guess execution latency, memory usage, or algorithm correctness.
- Always execute candidate programs through the in-process IPython harness.
- Microsecond timing, peak memory, and assertion passes are measured empirically.
- Evaluators set a `fitness = <float>` variable, or the engine computes runtime-penalized scores.

### 3. Context-as-a-Variable Token Shield
- Full candidate genomes (>1,500 chars) are stored out-of-band in the project SQLite vault (`prime_db.store_variable` or `openevolve-prime commit`).
- Only compact summaries (SHA-256 hash, generation, runtime in ms, fitness score) are kept in conversation context.
- Slashes evolutionary token consumption by 85–95%.

### 4. Per-Project SQLite Lineage & Pareto Frontier
- All evolutionary history, parent/child relationships, mutations, and metrics are preserved in `<project>/.prime-kernel/workspace.db`.
- State is 100% isolated per workspace—no cross-project contamination.

---

## ⚡ How to Operate OpenEvolve Prime in This IDE

Depending on the tools available in your current environment (Terminal, Python, or MCP), use one of the following methods:

### Method A: Using Terminal / CLI Commands (Recommended for Cursor, VS Code, Terminal)
Run `openevolve-prime` directly from your shell tool:

```bash
# 1. Initialize project engine (creates .prime-kernel/workspace.db & IDE rules)
openevolve-prime init

# 2. Empirically evaluate candidate code against a test harness
openevolve-prime eval --code "def solve(x): return x * 2" --eval "assert solve(5) == 10; fitness = 100.0"

# 3. Or evaluate files
openevolve-prime eval --code-file candidate.py --eval-file evaluator.py

# 4. Commit an individual to the evolutionary lineage
openevolve-prime commit --code-file candidate.py --fitness 98.5 --gen 1 --island island_0 --type ast_mutation

# 5. Inspect the current Pareto frontier
openevolve-prime frontier

# 6. View the all-time project champion
openevolve-prime champion

# 7. Fetch full code of an individual from the vault
openevolve-prime fetch --id ind_g1_abc123
```

### Method B: Using Python API (Inside Scripts or Interactive REPL)
```python
from prime_core import openevolve_prime

# Connects to current workspace engine (.prime-kernel/workspace.db)
engine = openevolve_prime.OpenEvolvePrimeEngine()

candidate_code = '''
def optimize_sort(arr):
    # Candidate sorting or algorithmic heuristic
    return sorted(arr)
'''

evaluator_code = '''
arr = [5, 2, 8, 1, 9]
assert optimize_sort(arr) == [1, 2, 5, 8, 9]
fitness = 100.0
'''

# 1. Microsecond in-process execution in IPython kernel
res = engine.evaluate_candidate(candidate_code, evaluator_code)
print(f"Status: {res['status']} | Fitness: {res['fitness']} | Latency: {res['runtime_ms']}ms")

# 2. Record to lineage
individual = engine.commit_individual(
    generation=1,
    mutation_type="seed_baseline",
    code=candidate_code,
    fitness_score=res["fitness"],
    metrics=res["metrics"],
    island_id="exploration_island_0"
)

# 3. Query Pareto frontier and champion
champ = engine.get_champion()
print("Champion Hash:", champ["genome_hash"])
```

### Method C: Using MCP Tools (OpenCode, Cursor, Claude Desktop, Windsurf, Zed, Cline)
Because OpenCode is globally configured with the OpenEvolve Prime MCP server, you can call these tools directly:
- `openevolve_evaluate_code(candidate_code, evaluator_code)`
- `openevolve_record_individual(code, fitness_score, generation, mutation_type, island_id)`
- `openevolve_get_frontier(limit, island_id)`
- `openevolve_get_champion(island_id)`
- `openevolve_fetch_code(individual_id)`
- `openevolve_get_instructions()`
- `openevolve_init_project(target_dir)`

---

## 🔄 Standard 5-Step Evolutionary Workflow

1. **Baseline Formulation**: Define the task, seed algorithm, and an objective evaluator harness with clear assertions.
2. **Generation Cohort Generation (Island 0)**: Mutate the seed structurally (novel AST nodes, different data structures, alternate heuristics).
3. **Persistent REPL Evaluation**: Run each candidate through `openevolve-prime eval`. Reject failed assertions immediately.
4. **Crossover & Refinement (Island 1)**: Combine the highest-scoring traits into hybrid candidates; optimize constants and loops.
5. **Pareto Verification & Lineage Commit (Island 2)**: Stress-test against boundary conditions; commit passing champions to `.prime-kernel/workspace.db`.