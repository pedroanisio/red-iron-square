# Above-Human-Level Agent Capabilities: Deep Research Report

> Research date: 2026-03-11
> Scope: Proposed additions to the personality-driven agent simulation framework

---

## Current Hard Ceilings (Honest Assessment)

The framework today is a **stimulus-response system with rich emotional telemetry**. Six architectural walls block superhuman capability:

| Ceiling | Why It's Fatal |
|---|---|
| Frozen true personality `psi` | Agent cannot learn, grow, or adapt |
| Emotions are read-only | Detected but never drive behavior |
| No goals or values | No agency — only reactive utility |
| Horizon = 1 tick | No planning, no strategy, no delay of gratification |
| Linear utility `Sigma(act_i * mod_i)` | No interaction effects, no nonlinear insight |
| No social cognition | Blind to other agents, no cooperation or competition |

Every proposal below targets one or more of these walls.

---

## Layer 1: Active Inference Backbone (Replaces Boltzmann)

**What:** Replace the Boltzmann softmax with Expected Free Energy (EFE) policy selection from Friston's active inference framework.

**Why superhuman:** Unifies exploration (epistemic value) and exploitation (pragmatic value) under a single principled objective. Humans do this intuitively but inconsistently — the math does it optimally.

**Concrete formula:**
```
G(pi) = -E_Q[H[P(o|s)]]    # ambiguity (explore to reduce)
        -E_Q[ln P(o)]        # risk (exploit preferred outcomes)
```

**Personality integration:** OCEAN maps to precision parameters:
- High N -> high interoceptive precision (amplifies emotional signals)
- High O -> low prior precision (accepts more surprise, explores more)
- High C -> high goal-prior precision (stays on task)

**Reference:** Champion, Bowman, Markovic & Grzes (2024). "Reframing the Expected Free Energy: Four Formulations and a Unification." Implementation available via [pymdp](https://github.com/infer-actively/pymdp).

---

## Layer 2: Metacognitive Control Loop

**What:** A MIDCA-style Note-Assess-Guide (NAG) monitor wrapping the decision cycle.

**Why superhuman:** Humans have metacognition but it's slow, biased, and inconsistent. A formal metacognitive loop operates at every tick with perfect self-access.

**Concrete mechanism:**
1. **Note** anomalies in the cognitive trace (entropy outside bounds, emotion oscillation, identity drift spike)
2. **Assess** severity against learned thresholds
3. **Guide** by adjusting: temperature, precision weights, learning rates, or triggering memory consolidation

**What this unlocks:** Self-regulating agent that knows *when it's confused* and adapts strategy. Bridges the "generation-verification gap" identified by Shakarian & Wei (2025).

**Reference:** Cox et al. (2016). "MIDCA: A Metacognitive, Integrated Dual-Cycle Architecture." Also: Toy, MacAdam & Tabor (2024). "Metacognition is all you need?" arXiv:2401.10910.

---

## Layer 3: Emotion Regulation Engine (Closes the Loop)

**What:** Implement Gross's Process Model as an active regulation layer, using the Petter et al. (2025) cost-benefit strategy selection.

**Why superhuman:** Humans regulate emotions but with systematic failures (rumination, suppression backfires, alexithymia). A computational agent can select the optimal regulation strategy per context.

**Five strategies as computational operations:**

| Strategy | Mechanism | When |
|---|---|---|
| Situation selection | Bias utility away from aversive scenarios | Proactive, low emotion |
| Attentional deployment | Reduce precision on triggering stimulus | Fast, high-intensity |
| Cognitive reappraisal | Re-run appraisal with alternate frame | Moderate intensity, durable |
| Response modulation | Scale behavioral expression, not internal state | Social contexts |
| Acceptance | Lower regulation effort, observe | When cost exceeds benefit |

**Personality modulation:** High O favors reappraisal (reframing). High C favors situation selection (avoidance planning). High N lowers regulation threshold (regulates more often but less effectively).

**Reference:** Petter, Mehta, Petrova, Kindt, Sheppes, Haslbeck & Gross (2025). "Emotion regulation, fast or slow." *Emotion*.

---

## Layer 4: Prospective Cognition (Mental Time Travel)

**What:** An Episodic Future Thinking module that constructs and evaluates hypothetical future scenarios by recombining episodic memories.

**Why superhuman:** Humans simulate futures but with massive availability bias, emotional contamination, and limited branching. An agent can systematically evaluate exponentially more futures.

**Concrete mechanism:**
1. Retrieve relevant episodic memories by similarity to current scenario
2. Recombine elements (people, contexts, outcomes) into 2-N future scenarios
3. Roll forward the temporal simulator on each branch (you already have `TemporalSimulator.tick()`)
4. Feed simulated future utilities back into current-tick EFE computation
5. Weight branches by personality: high N -> overweight threat branches; high O -> generate more diverse branches; high C -> longer horizon

**Reference:** Lee & Kwon (2024). "Episodic Future Thinking Mechanism for Multi-agent RL." arXiv:2410.17373. Also: Schacter & Addis (2007). "Constructive memory: remembering the past and imagining the future."

---

## Layer 5: Memory Architecture Overhaul

**What:** Replace the flat `MemoryBank` deque with a multi-system memory architecture.

**Current state:** One deque, rolling stats, no consolidation, no forgetting curve, no causal links.

| Memory System | Purpose | Concrete Mechanism |
|---|---|---|
| **Episodic** (exists) | What happened | Add Ebbinghaus forgetting: `retention = initial * exp(-lambda*t)`, causal tagging (action->outcome links) |
| **Semantic** (new) | What is true | Extract rules from episodic patterns: "high-stress scenarios with bold actions -> negative outcomes" |
| **Procedural** (new) | What works | Action-context success rates, evolving action modifiers |
| **Working Memory** (new) | What matters now | Limited-capacity active buffer (3-5 items) driving attention |
| **Prospective** (new) | What to do next | Active goals with deadline/priority/progress tracking |
| **Autobiographical** (new) | Who am I | Key events, turning points, narrative identity linking to self-model |

**Consolidation cycle:** After N ticks, run offline consolidation:
- Move high-valence episodic memories to long-term (emotional tagging)
- Extract semantic patterns from episodic clusters
- Update procedural records from outcome statistics
- Prune low-retention memories via forgetting curve

**Why superhuman:** Humans have these memory systems but with poor consolidation, interference, and retrieval failures. The computational version has perfect indexing and systematic consolidation.

---

## Layer 6: Bayesian Theory of Mind

**What:** Each agent maintains lightweight BToM models of other agents — priors over their personality vectors, updated from observed actions via inverse planning.

**Why superhuman:** Humans have ToM but it's slow (4+ years to develop), biased (egocentrism, projection), and limited to ~2 levels of recursion. Computational ToM can be precise, unbiased, and arbitrarily deep.

**Concrete mechanism (LAIP — LLM-Augmented Inverse Planning):**
1. Agent A observes Agent B's action in a scenario
2. A's model proposes candidate personality hypotheses for B
3. Bayesian update: `P(psi_B | action) ~ P(action | psi_B) * P(psi_B)`
4. Where `P(action | psi_B)` = Boltzmann probability under hypothesized personality
5. A uses posterior `psi_B` to predict B's future behavior

**Reference:** Gelpi, Xue & Cunningham (2024). "Towards Machine Theory of Mind with LLM-Augmented Inverse Planning." Also: Kim et al. (2025). "Hypothesis-Driven Theory-of-Mind Reasoning" (Sequential Monte Carlo approach).

---

## Layer 7: Curiosity-Driven Exploration

**What:** Add intrinsic motivation signals to the utility/EFE computation.

**Formula:**
```
r_intrinsic = alpha * novelty(s) + beta * info_gain(s,a) + gamma * empowerment(s)
```

**Personality modulation:**
- High O -> amplifies alpha (novelty-seeking)
- High C -> amplifies beta over alpha (systematic learning over random exploration)
- High N -> dampens gamma (avoids uncertainty about control)

**Why superhuman:** Humans have curiosity but it's easily hijacked (clickbait, addiction) and inconsistent (boredom kills exploration). A formal curiosity drive maintains optimal exploration pressure indefinitely.

**Reference:** Schmidhuber (2010). "Formal Theory of Creativity, Fun, and Intrinsic Motivation." Also: Burda et al. (2019). "Exploration by Random Network Distillation."

---

## Layer 8: Adaptive Expertise via Option Discovery

**What:** Model skill acquisition as hierarchical RL option discovery. Actions start primitive, then compress into chunks (macro-actions) through experience.

**Dreyfus stages as computational states:**
1. **Novice** — flat policy, explicit rule-following
2. **Competent** — learned value function, goal-directed planning
3. **Expert** — discovered options, automatic execution
4. **Adaptive Expert** — option recombination in novel contexts

**Personality modulation:**
- High C -> more deliberate practice (repetitions before promotion)
- High O -> more option recombination (adaptive expertise)
- Low O -> routine expertise (reliable execution, no recombination)

**Reference:** Sutton, Precup & Singh (1999). "Between MDPs and semi-MDPs: temporal abstraction in RL." Also: "Action Chunking in RL" (arXiv:2507.07969, 2025).

---

## Layer 9: Conceptual Creativity Engine

**What:** When facing novel problems, apply conceptual blending on episodic memories + MAP-Elites diversity search.

**Mechanism:**
1. Retrieve two dissimilar episodic memories
2. Project shared structure into a "blended space" (Fauconnier & Turner)
3. Evaluate candidate solutions via MAP-Elites (maintain personality-feature-indexed archive)
4. High O -> transformational blending (alter space rules); High C -> exploratory (thorough search)

**Why superhuman:** Humans blend concepts naturally but inconsistently and with severe fixation effects. A systematic blending + diversity search explores creative space exhaustively.

**Reference:** Mouret & Clune (2015). "Illuminating search spaces by mapping elites." Eppe et al. (2018). "A computational framework for conceptual blending."

---

## Integration Architecture

The nine layers form a coherent stack, not independent add-ons:

```
+---------------------------------------------+
|          METACOGNITIVE CONTROL (L2)         |  monitors everything,
|          Note -> Assess -> Guide            |  adjusts parameters
+---------------------------------------------+
|                                             |
|  PROSPECTION (L4)  <->  CREATIVITY (L9)    |  generate futures
|       |                      |              |  and novel options
|  +-------------------------------------+   |
|  |    ACTIVE INFERENCE ENGINE (L1)     |   |  unified decision
|  |    EFE = epistemic + pragmatic      |   |  framework
|  |         + curiosity (L7)            |   |
|  +-----------+--------------------------+   |
|              |                              |
|  EMOTION REGULATION (L3) <-> AFFECTIVE ENG |  bidirectional
|              |                              |
|  +-------------------------------------+   |
|  |    MULTI-SYSTEM MEMORY (L5)         |   |  episodic, semantic,
|  |    + consolidation + forgetting     |   |  procedural, working
|  +-----------+--------------------------+   |
|              |                              |
|  SKILL ACQUISITION (L8)  <->  ToM (L6)    |  learn + social
|                                             |
+---------------------------------------------+
|     PERSONALITY (psi) -- parameterizes ALL  |  precision weights,
|     SELF-MODEL (psi_hat) -- tracks ALL      |  learning rates,
|                                             |  thresholds
+---------------------------------------------+
```

---

## What Makes This "Above Human Level"

Humans have all nine capabilities but with systematic deficiencies:

| Capability | Human Limitation | Computational Advantage |
|---|---|---|
| Active inference | Inconsistent exploration/exploitation tradeoff | Optimal EFE computation |
| Metacognition | Slow, biased, Dunning-Kruger | Perfect self-access, every tick |
| Emotion regulation | Rumination, suppression failures | Optimal strategy selection |
| Prospection | Availability bias, limited branching | Exhaustive scenario evaluation |
| Memory | Interference, false memories, poor consolidation | Perfect indexing, systematic consolidation |
| Theory of Mind | Egocentric bias, ~2 levels recursion | Unbiased, arbitrary depth |
| Curiosity | Easily hijacked, inconsistent | Formal optimal exploration |
| Skill acquisition | 10,000 hours, plateau effects | Accelerated option discovery |
| Creativity | Fixation, conformity pressure | Systematic diversity search |

The key insight: **each human cognitive faculty has well-documented failure modes**. A computational implementation can preserve the *architecture* while eliminating the *failure modes*. That's what "above human level" means here — not different intelligence, but the same intelligence without the bugs.

---

## Suggested Implementation Order

1. **Memory overhaul (L5)** — foundational; everything else depends on richer memory
2. **Emotion regulation (L3)** — closes the biggest feedback gap in the current system
3. **Active inference (L1)** — replaces Boltzmann, unifies the math
4. **Prospection (L4)** — you already have `TemporalSimulator.tick()`, just run it forward
5. **Metacognition (L2)** — wraps around existing pipeline
6. **Curiosity (L7)** — add-on to the EFE computation
7. **ToM (L6)** — requires multi-agent infrastructure
8. **Skill acquisition (L8)** — requires action space expansion
9. **Creativity (L9)** — requires semantic memory + conceptual representations
