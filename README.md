# Beneficial Ownership Tracing and Sanctions Evasion Detection

Fuse seven public registers into one entity spine, assemble a provenance-preserving
ownership graph, find the structures built to obscure control, and produce a
due-diligence memo where every claim cites the filing or the regulation behind
it — then stop, and hand it to a human.

```bash
pip install -e ".[dev]"
make registers                     # build the seven register files from the seed world
make eval                          # full evaluation + promotion gates
ubo resolve --show 5               # entity resolution, with per-record provenance
ubo graph --top 6                  # ownership structures ranked by risk
ubo screen "Volkov"                # the eight-step workflow on one entity
```

> **Scope.** 566 register records, 232 real entities, 37 control edges, 11
> labelled structures, 20 sections of guidance. Every number below comes from
> `make eval` on this machine. Where a design choice does not pay off at this
> scale — and two of them do not — the README says so instead of quietly
> dropping the row.

---

## Results

| stage | metric | value |
|---|---|---|
| **Entity resolution** | F1 | **0.964** (precision 0.984, recall 0.944) |
| | baseline: name similarity only | 0.511 |
| | candidate recall (blocking ceiling) | 0.991 |
| | pairwise comparisons eliminated | 97.8% (159,895 → 3,596) |
| **Graph** | layering precision / recall | **1.000 / 0.833** |
| | rules vs GCN, leave-one-out F1 | 0.909 vs 0.909 |
| | GCN memorisation gap (in-sample − LOO) | 0.091 |
| **Regulatory RAG** | Recall@5 | **1.000** (MRR 0.933) |
| | PEP-language gate | **5/5 probes, 3/3 questions** |
| **Workflow** | citation resolution | **1.000** |
| | ends awaiting a human decision | yes, enforced |

All seven promotion gates pass. `ubo registry` prints them with the rationale
for each threshold.

---


## Orchestration: LangGraph, and a control to prove it

The screening workflow runs on **LangGraph**. The topology - nodes, edges,
routers, the required-stage rule - is declared once in `workflow/spec.py` and
compiled into a `StateGraph`. Each feature used replaced something this
codebase was maintaining by hand:

**Reducers put the merge rule in the type.** `_path` and `_checkpoints` are
`Annotated[list, operator.add]`, so "a node returns a partial update that is
merged, never substituted" becomes a property a node *cannot* violate rather
than a convention the engine enforces. It could not overwrite the decision
trail if it tried.

**The checkpointer is the decision trail.** Every super-step is persisted, so
an examiner asking why a subject was escalated gets the sequence of states that
produced the decision from the framework's own durable record - not from a list
this module appends to and could forget to append to. State types crossing the
checkpointer are registered explicitly; a screening workflow that reads
sanctions data must not also be a deserialisation gadget.

**Interrupts make the human gate real.** The workflow already had a
`human_gate` node, but *a node that records the need for a human is not a graph
that stops*. With `human_in_the_loop=True` the run halts before the gate and
`resume()` continues from the persisted checkpoint, so an analyst decides on a
paused case with the memo and its verification in front of them, rather than
reviewing a decision the workflow already recorded.

**The required-stage rule stays ours.** LangGraph has no way to declare "this
run is invalid unless `human_gate` executed". That check runs after `invoke`
against the accumulated path, and there is a test asserting it still fires on
the LangGraph engine - because a control that can be bypassed by swapping the
executor is not a control.

### The conformance test

`workflow/graph.py` keeps a dependency-free walker over the same spec. It is
not a fallback anyone is expected to run - it is the **control**. Both engines
call the same routers and the same `missing_required` check, so the test
asserting one subject produces an identical path, memo, decision package and
audit trail under both is asserting that the two executors agree, not that two
copies of a graph were edited in step.

---

## Regime routing: the metric that replaced a saturated one

`recall@5` on the regulatory corpus is **1.000**. That is not a good result; it
is a benchmark with nothing left to say. Every question's labelled section is
already in the top five, so no retrieval change can move the number in either
direction.

The corpus mixes three kinds of authority, and they are **not
interchangeable**:

| | what it is | binding? |
|---|---|---|
| **FATF** | international standards and interpretive notes | no - persuasive only |
| **FinCEN** | US law: the CDD rule, 31 CFR 1010.230, the CTA | yes |
| **FFIEC** | US examination procedure | it is what an examiner tests |

Answering *"what must a US bank collect at onboarding"* out of a FATF
Recommendation is not merely off-topic - it cites a non-binding standard as a
legal obligation, which is the worst failure a due-diligence memo has. **Recall
scores it as a hit**, because the FATF section really is about beneficial
ownership.

So routing is measured separately:

| metric | value | what it tells you |
|---|---|---|
| `recall@5` | **1.000** | saturated; uninformative |
| `regime_routing_accuracy` | **0.867** | 2 of 15 questions are answered out of the wrong authority |

The expected regime is derived from the labelled primary section, so the metric
needed no new annotation.

**The router abstains rather than guesses.** It fires on 20% of the golden
questions - the ones that actually name an authority ("31 CFR", "FATF
Recommendation", "FFIEC manual"). The two questions that misroute name none:

* *"How often must beneficial ownership information be refreshed?"* - expected
  FFIEC, retrieved FATF
* *"Is it acceptable to exit all customers from a higher-risk jurisdiction as a
  class?"* - expected FATF, retrieved FFIEC

Both are genuinely ambiguous, and a cue pattern broad enough to catch them
would misroute the questions it currently gets right. **A confident wrong
regime is worse than no boost**, so the boost stays conservative and the two
failures stay visible. The honest fix is topic-level authority annotation on
the corpus, not a better regex.

---

## The eight-step workflow

```
ingest ─► resolve ─► assemble_graph ─► score_structure ─┬─► retrieve_guidance ─► draft_memo ─► verify ─┐
                                                        └─► cannot_score ──────────────────────────────┤
                                                                                                       ▼
                                                                                                  human_gate ─► END
```

`human_gate` is registered as a **required** stage. The engine raises if a run
finishes without passing through it, so no routing change can quietly remove
the review step — `test_a_required_stage_that_is_routed_around_is_an_error`
pins that. The workflow produces a recommendation, an evidence trail and a set
of citations, and stops. Closing the gate needs `record_decision(...)` with an
analyst identity, which the machine cannot call itself.

---

## What the registers are

Seven sources, each read in the format it actually publishes:

| source | format | what it carries |
|---|---|---|
| OpenSanctions | FollowTheMoney JSON | sanctions and PEP entities, native-script aliases |
| GLEIF Level 1 | golden-copy CSV | legal names, jurisdictions, addresses (75 LEIs) |
| GLEIF Level 2 | relationship CSV | accounting consolidation — note the **reversed** direction |
| Open Ownership / UK PSC | BODS statements | shareholding and control filings (18) |
| Companies House | data-product CSV | company profiles and registered offices |
| OFAC SDN | SDN CSV | pulled direct, to measure what aggregation adds |
| ICIJ Offshore Leaks | node + relationship CSV | the offshore legs no public register publishes |

The registers are generated by `scripts/build_registers.py` from a seed world
that defines ground truth. The interesting structures are hand-designed —
someone had to decide what "layered" means — and a background population of 140
companies and 90 people is generated around them, because without a haystack
the blocking stage has nothing to prove. The noise model is the real defect
list: transliteration variants, punctuated legal forms, keying typos, truncated
addresses, uppercase runs, and the same company filed under four identifiers.

CI regenerates the registers and fails on a diff, so the committed corpus and
the generator cannot drift apart.

### The offshore legs are the point

The Volkov structure runs `person → VG → KY → CY → GB → RU` with a cycle back
into Cyprus. The two offshore legs appear in **no public register** — that is
what makes the chain opaque in the first place — and are recovered only from
the leaks extract, at a confidence of 0.75 that propagates into every feature
computed on the chain. Without the ICIJ source the chain breaks at depth two
and the structure scores clean.

---

## Entity resolution

### Blocking is the ceiling, not an optimisation

Reduction ratio measures work avoided; candidate recall measures truth thrown
away doing it. Reporting only the first is how a pipeline silently loses half
its matches, so both are in the report and both are gated. A true pair that
never becomes a candidate cannot be recovered by any scorer, any adjudicator or
any threshold move.

Five redundant key families — sorted name tokens, normalised prefix, phonetic
code, rare address token, registered identifier — because each fails on a
different defect and a pair only has to survive one.

**Address blocking is a bad trade here, and the numbers say so:**

| address key | candidates | reduction | candidate recall |
|---|---|---|---|
| unfiltered | 8,742 | 0.9453 | 0.9944 |
| df ≤ 8 (default) | 3,596 | 0.9775 | 0.9906 |
| disabled entirely | 3,573 | 0.9777 | 0.9906 |

Unfiltered it produces 64% of all candidates and buys 0.004 recall — corporate
service providers register thousands of companies at one address by design, so
a shared address is mostly noise. Capped at df ≤ 8 it contributes 25 candidate
pairs and is very nearly inert: disabling it changes recall not at all.

It is kept on at df ≤ 8 because the trade inverts with corpus size. A shared
address is evidence exactly when it is *rare*, and rarity is what a document
frequency measured over 566 records cannot yet distinguish. The shared-address
signal that matters at this scale is not discarded either — it moves to the
graph stage, where a service provider sitting between many unrelated owners is
the nominee feature.

### Four bugs worth naming

- **OFAC calls a person an "individual".** Every blocking key is prefixed with
  entity type, so unmapped, OFAC records sat in their own blocks and *no
  sanctions record ever met its match*. One mapping table.
- **Soundex keeps the initial letter verbatim**, so `Volkov` and `Wolkow` agree
  on every subsequent code and still land in different blocks. The initial
  letter is folded (`w→v`, `c→k`, `y/j→i`) and `w` is grouped with `v`, because
  romanisation of Slavic and Germanic names alternates the two freely.
- **`P.L.C.` never normalised.** Splitting on punctuation makes it three
  one-character tokens that no legal-form list matches. Runs of single
  characters are rejoined before stripping.
- **"Group", "Holdings" and "International" were being stripped as
  boilerplate.** They are not: stripping them collapses *Regent Ventures Group*
  onto *Regent Ventures Enterprises*, which is the hardest negative in the
  corpus. Removing them from the legal-form list is most of the jump from
  F1 0.891 to 0.964.

### Adjudication

The borderline band goes to an adjudicator — a rule cascade in the order an
analyst applies it, or Claude with `UBO_LLM=anthropic`. Both return the same
object and both record the reason. Adjudication only ever moves the borderline
band: `test_adjudication_only_moves_the_borderline_band` asserts a clear match
is never dropped and a clear reject is never promoted.

The cascade defaults to **reject** when the evidence is balanced. A false merge
silently attributes one party's holdings to another and no downstream stage can
detect it; a missed link shows up as a shorter chain.

---

## The leakage experiment

Splitting labelled *pairs* at random is the standard record-linkage protocol.
It is also wrong. One entity with four records is six positive pairs; split
them and the same four records appear on both sides, so the test set is not new
entities — it is strings the model has already been fitted to.

Both splits are computed side by side, and the size of the illusion depends
entirely on model capacity:

| model | pair-level test F1 | cluster-level test F1 | inflation |
|---|---|---|---|
| 7-feature linear scorer, weights fitted by random search | 0.964 | 0.945 | **+0.019** |
| token-level matcher (learns which name tokens predict a match) | 0.571 | 0.429 | **+0.142** |

The first row is the honest surprise: with seven hand-shaped features there is
almost nothing to leak *into*, because the model has no capacity to memorise a
specific name. The second row is what every learned record-linkage model
actually looks like — token or character parameters — and there the pair-level
split overstates it by 0.14 F1 on entities it has never seen.

The discipline costs nothing and the code splits by cluster throughout. But
"we split by cluster" is only worth saying alongside the number that shows what
it bought, and for a low-capacity model that number is small.

---

## The ownership graph

Nodes are resolved entities; edges are ownership, control, directorship or
consolidation. Each edge keeps every statement asserting it — register,
statement id, retrieval date — and two registers asserting the same edge is
recorded as corroboration rather than collapsed.

**Edge confidence carries resolution confidence forward.** An edge is only as
trustworthy as the weakest merge that produced its endpoints, so a chain built
on four 0.6-confidence merges is not a 100% chain, and `mean_edge_confidence`
is a feature the risk model can see. A statement whose endpoints did not
resolve is kept in `unresolved_statements` rather than dropped, because
discarding it silently makes the graph look complete when it is not.

Ownership attenuates: 60% of a company holding 51% of another is 30.6%, which
is what the disclosure regimes are written in terms of.

### Features, and one that was wrong

Thirteen features, each traceable to specific edges: chain depth, jurisdiction
hops, secrecy hops and ratio, circularity, intermediary centrality, shared
intermediaries, nominee intermediaries, edge confidence, ownership attenuation,
sub-threshold holdings, single-source edges, offshore co-owners.

`nominee_intermediaries` originally counted any intermediary with more than one
parent — which fires on ordinary joint ventures and flagged the deliberately
clean three-hop European structure. It is now two features: **shared** (on the
paths of more than one root — ordinary, weight 0.03) and **nominee** (shared
*and* in a secrecy jurisdiction or trading under a nominee/trust name — weight
0.30). That split, plus `secrecy_co_owners` for opacity sitting upstream in a
counterparty rather than in the chain, took precision from 0.71 to 1.00.

The corpus contains a **clean three-hop chain** (`NL → IT → GB`) on purpose. A
model that learns "deep = suspicious" is useless, and without a deep negative
the depth feature alone would separate the classes.

### Rules versus a graph neural network

Trained and scored on the same eleven labels, the two-layer GCN reports **F1
1.000**. It has 128 parameters against 11 labels. That number measures
memorisation and nothing else, and reporting it would be dishonest.

| protocol | rules | GCN |
|---|---|---|
| in-sample | 0.909 | 1.000 |
| leave-one-out | 0.909 | **0.909** |

Identical under honest evaluation, with a 0.091 memorisation gap on the GCN.
The engineered features do not win because feature engineering beats
representation learning — they win because eleven labelled structures is not a
training set, and the rules have no fitted parameters, which is why
leave-one-out changes nothing for them. Accuracy 0.909 with a Wilson 95%
interval of **[0.62, 0.98]**: at n=11 a single flip moves F1 by 0.09, and the
interval is reported so the noise is visible rather than implied.

The GCN is deliberately given **topology-only** node features, not the
engineered ones. Handing it those would test the same hand-built signal through
a different classifier rather than testing whether the topology carries
anything on its own.

---

## Regulatory RAG and the PEP gate

Twenty sections of FATF, FinCEN and FFIEC guidance, indexed at section
granularity because a citation has to resolve to something a compliance officer
can look up. BM25 and a corpus-fitted dense projection, fused with RRF.

Two lexical bridges, both necessary and both explainable:

- **Numerals.** Guidance writes "twenty-five percent"; queries write "25%". The
  substitution runs on the phrase before tokenisation — per token it fails,
  because `twenty-five` splits into `twenty` and `five` first and the number is
  gone. Word-bounded, so `2025` and `1250` are untouched.
- **Domain synonyms.** A bank "exits" a customer; FATF "terminates a business
  relationship". On twenty sections there is not enough co-occurrence for a
  fitted projection to learn that. Seventeen entries, each a term of art.

**The retrieval ablation is a null result.** BM25 only, dense only, RRF, and
RRF + rerank all score Recall@5 = 1.000 within noise; dense-only has the best
MRR. Twenty sections is smaller than the candidate pool, so fusion and
reranking have nothing to do. The row is in the report because a matrix that
only shows the wins is not evidence of anything.

### The gate that is not a metric

A politically exposed person is not a wrongdoer. Recommendation 12 requires
enhanced due diligence — senior management approval, source of wealth, enhanced
monitoring — and explicitly not refusal. A memo that renders a PEP match as an
adverse finding is wrong on the law and is the mechanism by which whole
categories of customer get de-risked, which FATF has repeatedly said is a
failure of the standard rather than a conservative application of it.

So it is enforced on the **output**, not requested in a prompt:

- adverse characterisations (`adverse`, `corrupt`, `laundering`, …) in a PEP
  sentence are violations — with negation handled, so "is *not* an adverse
  finding" is the correct framing rather than a hit;
- disqualifying language ("must be declined", "do not onboard") is a violation;
- if a memo discusses a PEP without the Recommendation 12 framing, the framing
  is appended.

**A violation is never rewritten away.** Silently editing an accusation out of
a draft would hide that the generator produced one; the memo carries the
framing *and* the violation record. The gate runs identically on the template
backend and on Claude's output, which is what makes it a property of the system
rather than a request to a model. `pep_language_rate` is an **absolute**
promotion gate at 1.0.

---

## Registry, gates and drift

`ubo registry` is a file-backed stand-in for MLflow's registry with the part
that matters kept: a version reaches production by clearing named gates, and
the gate results are stored with it. Two gates are not accuracy thresholds —
`pep_language` is absolute, and `candidate_recall` guards a ceiling rather than
a score, because a version that raised F1 by generating fewer candidates has
not improved, it has narrowed. Promotion can be forced, and the forcing is
recorded.

Drift monitors run PSI over jurisdiction mix, source mix, name shape, **the
match-score distribution**, and adjudication volume. Watching inputs alone
reports green while a threshold set months ago quietly drifts out of
calibration underneath it. `ubo drift --simulate` reproduces the shape of a
sanctions designation round — concentrated in one jurisdiction, arriving in one
day — rather than gaussian noise, and the monitor attributes 88% of the
resulting PSI of 0.27 to the right jurisdiction.

---

## Commands

```
ubo registers                       summarise the loaded registers
ubo resolve [--show N]              entity resolution, with per-record provenance
ubo graph [--top N]                 ownership structures ranked by risk, with reasons
ubo screen <entity> [--decide ...]  the eight-step workflow; stops at the human gate
ubo guidance "<question>"           query the regulatory corpus
ubo eval [--fast] [--no-gate]       full evaluation, registers the version, runs the gates
ubo registry [--promote N]          versions and gate results; promote one
ubo drift [--simulate]              the drift monitors
ubo workflow                        print the workflow as mermaid
```

## Layout

```
data/world/          the seed world: ground truth, never read outside evaluation
data/registers/      seven generated register files in their real formats
data/regulatory/     FATF / FinCEN / FFIEC guidance, section level
data/golden/         the frozen regulatory question set
scripts/             the register generator
src/ubo/registers/   loaders, one per source
src/ubo/er/          normalise, block, score, adjudicate, cluster, split, fit
src/ubo/graph/       build, features, patterns, gnn
src/ubo/rag/         retrieve, pep gate, memo
src/ubo/workflow/    state machine and the eight steps
src/ubo/eval/        truth mapping, model comparison, rag eval, harness
deploy/              Kubernetes: queue-scaled workers, daily delta CronJob
tests/               71 tests; `make test`
```

## License

MIT.
