# Parity with Goedel-Architect

Source: [arXiv:2606.06468](https://arxiv.org/abs/2606.06468), CC BY 4.0. Appendix C
reproduces the three system prompts; user prompts are omitted there.

Reported result for reference: **75.6% pass@1 on PutnamBench** with DeepSeek-V4-Flash
(284B-A13B) as the backbone, rising to 88.8% with a natural-language proof seeding the
blueprint.

## Adopted from the paper

`src/ax_prover/blueprint/prompts.py` follows Appendix C closely wherever the text does not
depend on their node encoding:

- The prover's task framing and tool discipline, including "the compiler is a stronger
  signal source than search" and the explicit warning not to use `mathlib_search` to find
  the proof. This directly targets a pathology we measured: 178 searches across 3 nodes.
- The architect's decomposition guidelines: at most 1-2 new logical ideas per lemma beyond
  its parents, split further if more, independent branches stay independent.
- The requirement that every statement be closed, typed, and standalone, with no reaching
  into ambient context, and that every proof sketch cite parents by backticked name and
  avoid "by algebra" / "obviously" / "one can check".
- Content-derived `snake_case` naming, topological declaration order.
- The refiner's two-bucket guidance, its instruction to re-read a suspect `PROOF_TOO_HARD`
  as `STATEMENT_WRONG`, and the rule that proved nodes carry forward while their signature
  is byte-identical.
- The in-band input format: `-- PROVED` / `-- UNPROVED` markers with `/- Diagnosis -/`
  review blocks, and the three-part review (`## Diagnosis`, `## Analysis`,
  `## Suggested Fix`).
- The prover's two-case compile contract: a `sorry` body compiles for exploration but
  cannot register a solve.
- `Compilation SUCCESSFUL. Validation SUCCESSFUL.` as the string both structural roles
  iterate until.

## Deliberate deviations

| Area | Paper | Here | Why |
| --- | --- | --- | --- |
| Node metadata | `@[blueprint (statement := ...) (proof := ...)]` | fenced `ax-blueprint` JSON in a docstring | The attribute needs their `Architect` Lean library |
| Dependencies | `sorry_using [p1, p2]`, Lean-verified | `parents` in JSON, metadata-declared | Same reason. Declared parents drive scheduling; elaborated deps stay observational |
| Placeholder | `sorry_using [...]`; bare `sorry` rejected | `by sorry` | Same reason |
| Definitions | `def`, `abbrev`, `structure`, `instance` permitted | Helper lemmas only | Milestone-one scope (plan section 19) |
| Target emission | Architect emits it; a pre-check compares the signature verbatim | Harness renders it from the user's file | Makes drift impossible rather than detected |
| Diagnosis values | `STATEMENT_WRONG`, `PROOF_TOO_HARD` | plus `BUDGET_EXHAUSTED`, `INFRASTRUCTURE_ERROR` | Starvation was being mislabelled as difficulty, misdirecting refinement |
| Node acceptance | Lean compile | compile plus `#print axioms` allow-list | A compiler-clean proof can still reach a `sorry` |

## Known parity gaps

- **No definition nodes.** A decomposition needing a helper `def` cannot be expressed, so
  such problems are out of reach. This is the largest gap.
- **Dependency graph is declared, not Lean-verified.** A node may cite a parent it does not
  use, or use a sibling it did not declare; isolation is enforced by scratch-module
  construction instead.
- **User prompts are ours.** Appendix C omits theirs, so the problem-specific inputs are
  independently written and untested against theirs.
- **Prompts are unmeasured.** No A/B against the previous wording, so any solve rate is a
  measurement of this adaptation, not a reproduction of the paper's.
