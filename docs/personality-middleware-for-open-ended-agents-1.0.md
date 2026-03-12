---
disclaimer: >
  No information in this document should be taken for granted.
  Any statement or premise not backed by a real logical definition
  or verifiable reference may be invalid, erroneous, or a hallucination.
  All mathematical formulas require independent verification against
  the cited primary sources before implementation.
title: "Personality Middleware for Open-Ended Agents: Bridging the Simulation Engine to Real-World Agent Behavior"
version: "1.0"
date: "2026-03-12"
status: "Architecture proposal — NOT YET IMPLEMENTED; extends personality-as-precision-landscape-1.0"
depends_on: "personality-as-precision-landscape-1.0.md"
---

# Personality Middleware for Open-Ended Agents

## 0. Purpose and scope

Red Iron Square's core — personality vectors, precision engines, EFE decomposition, constructed emotion, self-evidencing, and narrative generative models — implements a mathematically grounded simulation of personality-driven cognition. However, the current SDK boundary assumes a **closed-loop** world: finite predefined actions, manually injected scenarios, externally supplied outcomes.

This document specifies seven adapter layers that bridge the simulation engine to **open-ended, real-world agent behavior**. The goal is to embed a personality into any LLM-based agent — a coding assistant, a customer support bot, a research analyst — and have that personality causally influence how the agent thinks, decides, and communicates.

The architecture preserves the existing core untouched. Every component described here is an **additive adapter** that consumes the core's protocols and types.

### 0.1 Relationship to existing architecture

The current SDK flow:

```
Manual Scenario → DecisionEngine → Boltzmann Selection → DecisionResult (data)
                                                              ↓
                                                        TickResult (data)
```

The proposed middleware flow:

```
Real Task Context → ScenarioEncoder → Scenario
                                         ↓
                         PersonalityMiddleware.before_action()
                                         ↓
                              ┌─ DecisionEngine (unchanged)
                              ├─ PrecisionEngine (unchanged)
                              ├─ EFEEngine (unchanged)
                              └─ ConstructedAffectiveEngine (unchanged)
                                         ↓
                              BehavioralModulator → LLM parameters
                                         ↓
                                  External Agent acts
                                         ↓
                              OutcomeObserver → outcome float
                                         ↓
                         PersonalityMiddleware.after_action()
                                         ↓
                              State update, self-model update
```

### 0.2 Design constraints

1. **No core modifications.** Every existing module (`src/personality/`, `src/temporal/`, `src/precision/`, `src/efe/`, `src/constructed_emotion/`, `src/self_evidencing/`, `src/narrative/`) remains unchanged.
2. **Protocol-driven.** New components satisfy structural protocols, following the existing `DecisionEngineProtocol` / `System2RuntimeProtocol` pattern.
3. **Opt-in composition.** Each adapter layer is independently useful. An agent can use only `ScenarioEncoder` + `BehavioralModulator` without the full temporal loop.
4. **300 LOC per file.** All implementations respect the project gate.
5. **Testable in isolation.** Each adapter has a pure-function core that can be unit-tested without LLM calls.

---

## 1. ScenarioEncoder — mapping task context to dimension intensities

### 1.1 The gap

`Scenario` is an N-dimensional vector in `[0,1]^N` where each component represents stimulus intensity for a personality dimension. Today these vectors are hand-authored:

```python
scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="pitch_meeting")
```

An open-ended agent receives arbitrary task contexts: a user prompt, a code diff, a stack trace, a customer complaint. There is no mechanism to convert these into dimension intensities.

### 1.2 Theoretical grounding

In active inference, observations at any level generate prediction errors weighted by precision. The scenario vector `s` in Red Iron Square serves as the **exteroceptive observation** — the external stimulus that the agent must respond to. The activation functions `f_i(s_i, θ_i)` compute the personality-modulated response to each stimulus dimension.

For open-ended usage, we need a function:

```
encode: TaskContext → s ∈ [0,1]^N
```

This is formally a **recognition model** (encoder) in the variational inference sense — it maps raw observations to the latent space that the generative model (personality dynamics) operates over.

### 1.3 Architecture

```
TaskContext (free text, structured metadata, or both)
       │
       ▼
┌────────────────────────┐
│   ScenarioEncoder      │
│                        │
│  ┌──────────────────┐  │
│  │ HeuristicEncoder │  │  ← Rule-based fast path (System 1)
│  └────────┬─────────┘  │
│           │ confidence  │
│           │ < threshold │
│  ┌────────▼─────────┐  │
│  │   LLMEncoder     │  │  ← LLM-based slow path (System 2)
│  └────────┬─────────┘  │
│           │             │
│  ┌────────▼─────────┐  │
│  │  CachedEncoder   │  │  ← Memoization layer
│  └──────────────────┘  │
└────────────┬───────────┘
             ▼
        Scenario object
```

### 1.4 Dimension encoding semantics

Each personality dimension maps to a stimulus axis with clear real-world semantics:

| Dimension | Stimulus axis | Low (0.0) | High (1.0) | Examples |
|-----------|--------------|-----------|------------|----------|
| **O** (Openness) | Novelty demand | Routine, well-defined task | Ambiguous, creative, exploratory | Refactoring (0.2) vs. greenfield design (0.9) |
| **C** (Conscientiousness) | Structure demand | Informal, exploratory | Strict requirements, compliance | Brainstorm (0.1) vs. production deploy (0.95) |
| **E** (Extraversion) | Social exposure | Solo, async work | Public, collaborative, high-visibility | Code review (0.3) vs. live demo (0.9) |
| **A** (Agreeableness) | Conflict potential | Cooperative, aligned | Contested, adversarial | Pair programming (0.2) vs. design disagreement (0.85) |
| **N** (Neuroticism) | Stress intensity | Low stakes, reversible | High stakes, irreversible, time pressure | Local test (0.1) vs. production incident (0.95) |
| **R** (Resilience) | Adversity level | Smooth execution | Repeated failures, blockers | Green CI (0.1) vs. flaky tests + deadline (0.9) |
| **I** (Idealism) | Ideal vs. pragmatic tension | Pure pragmatic context | Strong ideal/principled context | Quick fix (0.1) vs. architecture decision (0.85) |
| **T** (Tradition) | Convention pressure | Greenfield, no precedent | Strong existing patterns, standards | New project (0.1) vs. legacy codebase (0.9) |

### 1.5 HeuristicEncoder specification

The heuristic encoder uses keyword/pattern matching for sub-millisecond encoding without LLM calls. It operates as a **bag-of-signals** model:

```python
class HeuristicEncoder(Protocol):
    def encode(self, context: TaskContext) -> tuple[Scenario, float]:
        """Return scenario and confidence in [0, 1]."""
        ...
```

Signal sources:
- **Keywords** in the task description (e.g., "urgent", "production", "creative")
- **Metadata** (file types, git branch names, CI status, error counts)
- **Structural signals** (prompt length, number of constraints, presence of stack traces)
- **Historical context** (how many retries, time elapsed, previous outcomes)

Each signal contributes a partial dimension update. Signals are aggregated via weighted averaging with saturation clipping to `[0, 1]`.

Confidence is computed as `coverage / N` — the fraction of dimensions that received at least one signal. When confidence falls below a configurable threshold (default 0.6), the LLM encoder is invoked.

### 1.6 LLMEncoder specification

The LLM encoder uses a structured output call (matching the existing `AgentRuntime` pattern) to produce dimension values:

```python
class ScenarioProposalFromContext(BaseModel):
    """LLM-generated scenario encoding from task context."""
    values: dict[str, float]  # dimension code → [0, 1]
    name: str
    description: str
    rationale: str
```

The prompt template:

```
Given the following task context, estimate the stimulus intensity
for each personality dimension on a scale of [0, 1].

Dimensions:
- O (Openness): How much novelty/creativity does this task demand?
- C (Conscientiousness): How much structure/precision is required?
- E (Extraversion): How socially exposed or collaborative is this?
- A (Agreeableness): How much conflict potential exists?
- N (Neuroticism): How stressful or high-stakes is this?
- R (Resilience): How much adversity or failure is present?
- I (Idealism): How much ideal-vs-pragmatic tension?
- T (Tradition): How much pressure to follow conventions?

Task context:
{context}
```

This reuses the existing `StructuredLLMAdapter` protocol and `AnthropicAdapter`/`OpenAIAdapter` implementations. The LLM call is the **System 2 slow path** — invoked only when the heuristic encoder's confidence is insufficient.

### 1.7 CachedEncoder layer

A content-addressable cache (keyed by a hash of the task context) stores previously computed encodings. Cache policy: LRU with configurable max size (default 1024). This prevents redundant LLM calls for similar task contexts.

The cache key uses semantic hashing: the task context is normalized (lowercased, stop-words removed, sorted) before hashing. This allows slight variations in phrasing to hit the same cache entry.

---

## 2. ActionEncoder — mapping open-ended actions to modifier vectors

### 2.1 The gap

`Action` requires an explicit `np.ndarray` of modifiers with shape `(registry.size,)` — one float in `[-1, 1]` per personality dimension, authored at construction time:

```python
action = sdk.action("bold_pitch", {"O": 0.8, "C": -0.3, "E": 0.9, ...})
```

An open-ended agent's action space is unbounded: tool calls, API requests, free-text responses, code edits. There is no mechanism to produce modifier vectors for novel actions.

### 2.2 Theoretical grounding

In the active inference framework, the **action model** specifies how actions change the state of the world. The modifier vector `m ∈ [-1, 1]^N` represents the personality-alignment profile of an action: positive modifiers indicate the action is consonant with high values of that dimension; negative modifiers indicate dissonance.

For open-ended usage, we need:

```
encode_action: ActionDescription → m ∈ [-1, 1]^N
```

This is the **generative model's likelihood mapping** — it specifies the expected personality-alignment signature of an action.

### 2.3 Action archetype library

Rather than encoding every unique action from scratch, we define a library of **action archetypes** — prototypical behavioral patterns with pre-computed modifier vectors. Novel actions are classified to their nearest archetype(s).

| Archetype | O | C | E | A | N | R | I | T |
|-----------|-----|-----|-----|------|------|------|------|------|
| **Explore** (novel approach, creative solution) | +0.9 | -0.3 | +0.2 | 0.0 | -0.2 | +0.3 | +0.4 | -0.6 |
| **Systematic** (methodical, step-by-step) | -0.2 | +0.9 | -0.1 | +0.1 | -0.3 | +0.2 | +0.3 | +0.7 |
| **Collaborate** (seek input, delegate) | +0.1 | +0.2 | +0.8 | +0.7 | -0.2 | +0.1 | +0.1 | +0.2 |
| **Challenge** (push back, question) | +0.3 | +0.1 | +0.5 | -0.8 | +0.3 | +0.5 | +0.6 | -0.4 |
| **Cautious** (validate, double-check) | -0.3 | +0.7 | -0.2 | +0.2 | +0.5 | -0.1 | +0.3 | +0.5 |
| **Retry** (persist, try again) | 0.0 | +0.4 | 0.0 | 0.0 | +0.2 | +0.9 | +0.2 | +0.1 |
| **Escalate** (ask for help, raise flag) | -0.1 | +0.3 | +0.4 | +0.5 | +0.4 | -0.6 | 0.0 | +0.3 |
| **Pragmatic** (quick fix, good enough) | -0.2 | -0.2 | 0.0 | +0.1 | -0.3 | +0.2 | -0.8 | -0.3 |
| **Principled** (best practice, clean design) | +0.3 | +0.6 | 0.0 | +0.1 | -0.1 | +0.2 | +0.9 | +0.5 |
| **Withdraw** (disengage, defer) | -0.4 | 0.0 | -0.7 | +0.2 | +0.3 | -0.8 | 0.0 | 0.0 |

### 2.4 Classification mechanism

A novel action is classified to archetype(s) via:

1. **Keyword matching** (fast path): The action description is scanned for archetype-associated keywords. If a single archetype matches with high confidence, its modifier vector is returned directly.

2. **Soft classification** (when ambiguous): The action is projected onto multiple archetypes with weights `w_k ∈ [0, 1]`, and the modifier vector is a weighted average:

```
m = Σ_k w_k · m_k / Σ_k w_k
```

3. **LLM classification** (fallback): For truly novel actions, a structured LLM call classifies the action into archetype weights. This uses the same `StructuredLLMAdapter` infrastructure.

### 2.5 Extension: learned encoders

For agents operating in a stable domain (always coding, always doing customer support), the archetype weights can be learned from behavioral data. After N observed actions with known outcomes, a lightweight regression model maps action features to modifier vectors directly, bypassing both heuristics and LLM calls.

This connects to the **meta-learning objective** already implemented in `src/meta/`: the `CMAESOptimizer` could optimize archetype modifier vectors to maximize behavioral divergence (the existing `MetaObjective`), ensuring that personality differences produce measurably different action preferences.

---

## 3. OutcomeObserver — deriving outcomes from real-world signals

### 3.1 The gap

`TemporalSimulator.tick()` requires `outcome: float | None`. When `None`, outcome is auto-generated from utility scores — a synthetic feedback loop. For real agents, outcome must come from the actual result of the action.

### 3.2 Theoretical grounding

In active inference, the **outcome** is the observation that follows an action — it generates the prediction error that drives learning. The outcome's sign and magnitude determine how the agent's internal state evolves:

- Positive outcome → mood gain, satisfaction increase, frustration decay
- Negative outcome → mood loss (amplified by Neuroticism), frustration spike (damped by Resilience)

The state transition equations in `update_state()` and `update_state_precision()` already handle arbitrary outcome floats correctly. The gap is only in **sourcing** the float.

### 3.3 Signal taxonomy

Real-world outcomes are multi-dimensional. The `OutcomeObserver` aggregates multiple signals into a single `[-1, 1]` scalar:

| Signal category | Source | Mapping to outcome |
|----------------|--------|-------------------|
| **Task completion** | Did the action achieve its goal? | Binary: success → +0.5, failure → -0.5 |
| **Quality feedback** | User rating, edit distance, test results | Continuous: normalized to [-1, 1] |
| **Efficiency** | Time taken vs. expected, token cost | Deviation from expectation → [-0.3, +0.3] |
| **Error signals** | Exceptions, validation failures, lint errors | Count-based: -0.1 per error, capped at -0.8 |
| **Social signals** | User tone, explicit praise/criticism | Sentiment analysis → [-0.5, +0.5] |
| **Self-assessment** | Agent's own confidence in its output | Calibrated probability → [-0.3, +0.3] |

### 3.4 Aggregation formula

```
outcome = Σ_j w_j · signal_j / Σ_j w_j
```

Where `w_j` are configurable per-signal weights. Default weights prioritize task completion (0.4) and quality feedback (0.3), with efficiency (0.1), errors (0.1), and social (0.1) as secondary signals.

The aggregation is **personality-modulated**: a high-N agent amplifies error signals (weights errors higher), while a high-A agent amplifies social signals. This creates a feedback loop where personality influences which outcomes the agent attends to, which in turn shapes its state evolution — consistent with the precision-weighted prediction error framework in `personality-as-precision-landscape-1.0.md`.

### 3.5 Outcome precision

The `PrecisionEngine` already computes Level-0 interoceptive precision from personality and state. The `OutcomeObserver` can additionally report **outcome confidence** — how certain we are about the outcome signal. This maps directly to the observation precision in hierarchical active inference:

```
ε_weighted = Π_obs · (outcome_observed - outcome_predicted)
```

High-confidence outcomes (clear success/failure) produce sharp prediction errors. Low-confidence outcomes (ambiguous results) produce attenuated errors. This prevents noisy feedback from destabilizing the agent's internal state.

---

## 4. BehavioralModulator — translating internal state to agent behavior

### 4.1 The gap

The SDK produces `DecisionResult` and `TickResult` — rich data structures containing probabilities, utilities, activations, emotions, precision states, and self-model metrics. But none of this data **does** anything. It sits in memory while the agent continues to behave identically regardless of personality.

### 4.2 Theoretical grounding

In active inference, the agent's actions are selected to minimize expected free energy. The **policy** — the probability distribution over actions — is shaped by precision at Level 1 (policy precision γ). Higher γ means more confident, more deterministic action selection; lower γ means more exploratory, more stochastic.

Red Iron Square already computes γ via the `PrecisionEngine`. The missing step is mapping γ and other internal signals to the actual control parameters of an LLM-based agent.

### 4.3 Modulation targets

An LLM-based agent has several behavioral knobs:

| Target | Controlled by | Mechanism |
|--------|--------------|-----------|
| **Sampling temperature** | Policy precision γ (Level 1) | `T_llm = T_base / γ` |
| **System prompt overlay** | Personality vector θ, emotional state | Trait-specific instruction injection |
| **Response length bias** | Energy, Extraversion | Low energy → shorter responses |
| **Tool selection weights** | Action archetype probabilities | Personality-preferred tools weighted higher |
| **Retry policy** | Resilience, frustration | High R + low frustration → more retries |
| **Escalation threshold** | Neuroticism, energy | High N + low energy → escalate sooner |
| **Risk appetite** | Openness, Conscientiousness | High O + low C → accept more risk |
| **Tone** | Mood, Agreeableness | Mood < 0 + high A → diplomatic caution |

### 4.4 Temperature modulation

The most direct and mathematically grounded modulation. The `PrecisionEngine` already computes Level-1 policy precision γ. The temporal simulator already uses `1/γ` as temperature. We extend this to the LLM:

```
T_llm = clamp(T_base / γ, T_min, T_max)
```

Where `T_base` is the agent framework's default temperature, and `T_min`/`T_max` are safety bounds (default 0.1 and 1.5). This means:
- High precision (confident state, familiar task) → low temperature → deterministic outputs
- Low precision (uncertain state, novel task) → high temperature → exploratory outputs

This is not a metaphor. It is a direct application of the same Boltzmann temperature that the `DecisionEngine` already uses, extended to the LLM's sampling distribution.

### 4.5 System prompt overlay

The personality vector and current emotional state are rendered as a **system prompt fragment** injected before the agent's task-specific prompt:

```python
class SystemPromptOverlay(Protocol):
    def render(
        self,
        personality: PersonalityVector,
        state: AgentState,
        emotions: list[EmotionReading],
    ) -> str:
        """Generate personality-aware system prompt fragment."""
        ...
```

The overlay is **not** a static personality description. It is dynamically generated from the current state:

```
You approach problems with [O-derived creativity level].
You value [C-derived structure preference].
You are currently feeling [emotion labels from constructed affect].
Your energy level is [energy state] — [energy-derived instruction].
Your confidence in this domain is [precision-derived level].
```

This creates a **causal loop**: personality shapes the prompt, the prompt shapes LLM behavior, the behavior generates outcomes, outcomes update internal state, state changes the next prompt.

### 4.6 Action guidance

The `BehavioralModulator` doesn't replace the agent's decision-making — it **biases** it. The output is an `ActionGuidance` structure:

```python
class ActionGuidance(BaseModel):
    """Personality-derived behavioral guidance for the next action."""
    temperature: float
    system_prompt_overlay: str
    preferred_archetypes: list[str]  # ranked by personality fit
    risk_tolerance: float            # [0, 1]
    verbosity_bias: float            # [-1, 1]
    retry_budget: int                # max retries before escalation
    escalation_urged: bool           # true if state suggests escalation
```

The consuming agent framework can use as much or as little of this guidance as it wants. A minimal integration uses only `temperature`; a full integration uses all fields.

---

## 5. EventDrivenAdapter — lifecycle hooks for asynchronous agents

### 5.1 The gap

The temporal simulator assumes a synchronous tick loop:

```python
for scenario, outcome in zip(scenarios, outcomes):
    result = simulator.tick(scenario, outcome)
```

Real agents are event-driven: tasks arrive asynchronously, actions have variable duration, idle periods exist between tasks. The tick-based model doesn't map to this lifecycle.

### 5.2 Theoretical grounding

In active inference, the agent operates on a **generative process** that unfolds in continuous time. The discrete-time approximation (ticks) is valid when ticks align with meaningful state transitions. For real agents, meaningful transitions occur at:

1. **Task arrival** — new exteroceptive observation
2. **Action completion** — outcome observation
3. **Idle decay** — state relaxation toward equilibrium
4. **Error/interrupt** — unexpected perturbation

The temporal simulator's state transition equations already model decay (mood_decay, energy_decay, frustration_decay). The adapter maps real-world events to appropriate tick invocations.

### 5.3 Event protocol

```python
class AgentLifecycleEvent(Protocol):
    """Events in an agent's real-world lifecycle."""
    ...

class EventDrivenAdapter:
    """Maps agent lifecycle events to temporal simulation ticks."""

    def on_task_received(self, context: TaskContext) -> ActionGuidance:
        """New task arrives. Encode scenario, compute guidance."""
        ...

    def on_action_completed(
        self, context: TaskContext, result: ActionResult
    ) -> None:
        """Action finished. Observe outcome, update state."""
        ...

    def on_idle(self, elapsed_seconds: float) -> None:
        """No task active. Apply decay, recover energy."""
        ...

    def on_error(self, error: Exception, context: TaskContext) -> None:
        """Unexpected failure. Spike frustration, update precision."""
        ...

    def on_feedback(self, feedback: UserFeedback) -> None:
        """Explicit user feedback. Strong outcome signal."""
        ...
```

### 5.4 Idle decay mechanics

Between tasks, the agent's state should relax toward equilibrium. The existing `StateTransitionParams` defines decay rates (mood_decay=0.92, energy_decay=0.90, etc.). The adapter applies these as:

```
state_i(t + Δt) = equilibrium_i + (state_i(t) - equilibrium_i) · decay_i^(Δt/tick_duration)
```

Where `tick_duration` is a configurable time constant (default 1 second). This ensures that a 60-second idle period applies 60 ticks worth of decay, bringing mood toward 0.0, energy toward 0.8, and frustration toward 0.0.

### 5.5 Multi-tick actions

Some agent actions span multiple "ticks" of real time (e.g., a long code generation, a multi-step API call). The adapter supports **sub-tick state updates** — intermediate observations that update state without completing a full tick cycle. This prevents the agent from accumulating unrealistic amounts of energy recovery during a long action.

---

## 6. BehavioralEvidenceExtractor — grounding self-model in real behavior

### 6.1 The gap

The `SelfModelSimulator` in `src/self_model/model.py` updates `psi_hat` (the agent's internal estimate of its own personality) using **behavioral evidence** derived from probability-weighted action modifiers:

```
B_i(t) = Σ_k P(a_k) · m_k_i
```

This is synthetic — the agent's "behavior" is just the action probabilities from its own decision engine. For a real agent, behavioral evidence should come from **actual observed actions**.

### 6.2 Theoretical grounding

Self-evidencing in active inference (Friston, 2024; Laukkonen, Friston & Chandaria, 2025) posits that the agent maintains a generative model of itself — a self-model that predicts its own behavior. The self-model updates when behavior diverges from prediction, creating a feedback loop between identity and action.

In Red Iron Square, `psi_hat` is this self-model. The `SelfEvidencingModulator` modulates policy precision based on the divergence between predicted and actual action distributions. The missing piece is that "actual actions" in the current system are still model-generated.

### 6.3 Evidence extraction

The `BehavioralEvidenceExtractor` maps observed agent behavior to dimension-level evidence:

| Observed behavior | Dimension evidence |
|-------------------|-------------------|
| Chose an unconventional approach | O evidence: high |
| Followed coding standards precisely | C evidence: high |
| Communicated proactively | E evidence: high |
| Accommodated user's preference over own | A evidence: high |
| Added extra validation / error handling | N evidence: high |
| Retried after failure without escalating | R evidence: high |
| Chose best-practice over quick fix | I evidence: high |
| Followed existing patterns in codebase | T evidence: high |

The extractor uses the **ActionEncoder** (section 2) in reverse: the action that the agent actually took is classified into archetype weights, and the archetype's modifier vector becomes the behavioral evidence.

```
B_actual(t) = Σ_k w_k(actual_action) · m_k
```

This replaces the synthetic `B(t) = Σ_k P(a_k) · m_k` with evidence grounded in real behavior.

### 6.4 Self-model divergence and personality drift

With real behavioral evidence, the self-model can genuinely diverge from the true personality. This enables a phenomenon absent from the closed-loop simulation: **personality drift through environmental pressure**.

An agent with high Openness personality but deployed in a highly constrained environment (always following strict procedures) will accumulate Conscientiousness-aligned behavioral evidence. Over time, `psi_hat` drifts toward higher C, even though the true personality θ remains unchanged. This creates:

1. **Coherence gap** increase: `||psi_hat - B|| / √N` grows
2. **Identity drift**: `||psi_hat - psi_hat_0|| / √N` grows
3. **Constructed emotion**: The self-evidencing modulator detects the drift and may trigger a System 2 narrative refresh — the agent "notices" that it has been acting out of character

This is a testable prediction about personality-environment interaction, grounded in the self-evidencing mathematics already implemented.

### 6.5 Implications for the self-evidencing modulator

The `SelfEvidencingModulator` computes per-action precision weights:

```
Π_self(a) = Π_base · min(Π_max, exp(-β · d(a, psi_hat)))
```

With real behavioral evidence, the predicted probability distribution `P_hat` used to compute divergences `d(a, psi_hat)` becomes genuinely predictive — it reflects the self-model's expectation of how the agent will behave, not a tautological echo of the decision engine.

This means self-evidencing can now detect **genuine surprises**: the agent acts in a way its self-model didn't predict, triggering precision adjustment and potentially a System 2 narrative refresh.

---

## 7. AgentMiddleware protocol — integration with external frameworks

### 7.1 The gap

The SDK's protocols (`DecisionEngineProtocol`, `System2RuntimeProtocol`) are internal — they specify contracts between Red Iron Square components. There is no protocol for wrapping an external agent framework (LangChain, CrewAI, Claude Agent SDK, AutoGen, or a custom agent).

### 7.2 Architecture

The `PersonalityMiddleware` is the public integration surface. It composes all six previous adapters into a single coherent interface:

```
PersonalityMiddleware
  ├── ScenarioEncoder      (section 1)
  ├── ActionEncoder         (section 2)
  ├── OutcomeObserver       (section 3)
  ├── BehavioralModulator   (section 4)
  ├── EventDrivenAdapter    (section 5)
  ├── BehavioralEvidenceExtractor (section 6)
  │
  └── Core (unchanged)
      ├── AgentSDK
      ├── TemporalSimulator
      ├── PrecisionEngine
      ├── EFEEngine
      ├── ConstructedAffectiveEngine
      ├── SelfEvidencingModulator
      └── NarrativeGenerativeModel
```

### 7.3 Protocol specification

```python
class PersonalityMiddleware(Protocol):
    """Public integration surface for personality-aware agents."""

    def before_action(self, context: TaskContext) -> ActionGuidance:
        """Called before the agent acts.

        Encodes the task context as a scenario, runs the personality
        engine, and returns behavioral guidance (temperature, prompt
        overlay, preferred archetypes, risk tolerance, etc.).
        """
        ...

    def after_action(
        self,
        context: TaskContext,
        action_taken: ActionDescription,
        result: ActionResult,
    ) -> PersonalityState:
        """Called after the agent acts.

        Observes the outcome, extracts behavioral evidence, updates
        the temporal state and self-model, and returns the new
        personality state snapshot.
        """
        ...

    def on_idle(self, elapsed_seconds: float) -> PersonalityState:
        """Called during idle periods.

        Applies state decay and returns updated state.
        """
        ...

    def get_system_prompt_overlay(self) -> str:
        """Current personality-aware system prompt fragment."""
        ...

    def get_sampling_params(self) -> SamplingParams:
        """Current personality-derived sampling parameters."""
        ...

    @property
    def state(self) -> AgentState:
        """Current internal state (mood, energy, etc.)."""
        ...

    @property
    def emotions(self) -> list[EmotionReading]:
        """Currently active emotions."""
        ...

    @property
    def self_model(self) -> dict[str, float]:
        """Current self-model estimate psi_hat."""
        ...
```

### 7.4 Integration patterns

#### Pattern A: Minimal (temperature only)

The simplest integration — personality only affects sampling temperature:

```python
middleware = build_personality_middleware(
    personality={"O": 0.8, "C": 0.4, "N": 0.3, ...}
)

# Before each LLM call:
guidance = middleware.before_action(TaskContext(description=user_prompt))
response = llm.complete(prompt, temperature=guidance.temperature)

# After each LLM call:
middleware.after_action(context, action_taken, result)
```

#### Pattern B: Prompt overlay

Personality shapes the system prompt:

```python
guidance = middleware.before_action(context)
system_prompt = base_system_prompt + "\n" + guidance.system_prompt_overlay
response = llm.complete(system_prompt + user_prompt, temperature=guidance.temperature)
```

#### Pattern C: Full integration (Claude Agent SDK)

Complete personality-aware agent with lifecycle hooks:

```python
class PersonalityAwareAgent:
    def __init__(self, middleware: PersonalityMiddleware, agent: BaseAgent):
        self.middleware = middleware
        self.agent = agent

    async def run(self, task: str) -> str:
        guidance = self.middleware.before_action(
            TaskContext(description=task)
        )
        self.agent.temperature = guidance.temperature
        self.agent.system_prompt += guidance.system_prompt_overlay

        result = await self.agent.run(task)

        self.middleware.after_action(
            TaskContext(description=task),
            ActionDescription(text=str(result)),
            ActionResult(success=True, output=str(result)),
        )
        return result
```

#### Pattern D: Multi-agent ensemble

Different personalities for different agent roles:

```python
analyst = build_personality_middleware(
    personality={"O": 0.3, "C": 0.9, "N": 0.6, "I": 0.8}
)  # Careful, structured, detail-oriented

creative = build_personality_middleware(
    personality={"O": 0.95, "C": 0.2, "N": 0.1, "I": 0.4}
)  # Bold, exploratory, risk-tolerant

reviewer = build_personality_middleware(
    personality={"O": 0.4, "C": 0.8, "A": 0.3, "T": 0.7}
)  # Convention-following, critical, thorough
```

---

## 8. Theoretical extensions

### 8.1 Personality as attention allocation

The precision-as-personality thesis from `personality-as-precision-landscape-1.0.md` gains new empirical grounding in the open-ended setting. When the `ScenarioEncoder` produces dimension intensities and the `PrecisionEngine` weights prediction errors, the personality effectively determines **what the agent pays attention to**:

- High-N agent: amplifies stress signals in the scenario encoding → notices risks first
- High-O agent: amplifies novelty signals → notices creative opportunities first
- High-C agent: amplifies structure signals → notices missing specifications first

This is not a programmed heuristic — it emerges from the precision-weighted prediction error computation already in the core. The middleware simply provides real-world observations for the precision engine to weight.

### 8.2 Constructed emotion in real agents

The `ConstructedAffectiveEngine` currently generates emotions from free energy changes and precision-weighted prediction errors. In the open-ended setting, these emotions become **functionally real**:

- **Frustration** (high free energy, repeated negative outcomes) → changes behavior (more retries, lower temperature, eventual escalation)
- **Excitement** (high positive surprise, high arousal) → changes behavior (more exploration, higher temperature, longer responses)
- **Anxiety** (high N × high stress scenario × low energy) → changes behavior (more validation, more cautious tool selection)

The emotion doesn't need to be "felt" — it needs to be **functional**. The constructed emotion framework provides exactly this: emotions as functional states that modulate precision and action selection.

### 8.3 Identity through self-evidencing

The most theoretically novel aspect of the open-ended extension is **emergent identity**. In the closed-loop simulation, identity (psi_hat) is interesting but abstract. In a real agent:

1. The agent develops a behavioral pattern (e.g., always choosing careful, well-tested approaches)
2. The `BehavioralEvidenceExtractor` captures this pattern as dimension-level evidence
3. `psi_hat` updates to reflect the observed behavioral tendencies
4. The `SelfEvidencingModulator` uses `psi_hat` to predict future behavior
5. When the agent acts **out of character** (e.g., takes a risky shortcut under time pressure), the prediction error triggers precision adjustment
6. If the divergence is large enough, `System2Orchestrator` triggers a narrative refresh — the agent's "self-story" updates

This creates a genuine sense of **behavioral consistency** — not because the agent is programmed to be consistent, but because self-evidencing precision modulation makes consistency the path of least free energy.

### 8.4 Personality transfer and calibration

The middleware enables a new capability: **personality calibration from behavioral data**. Given a log of an agent's actions and outcomes, the `BehavioralEvidenceExtractor` can reconstruct the behavioral evidence stream. Fitting a personality vector θ that minimizes the divergence between predicted and observed behavior is a well-defined optimization problem:

```
θ* = argmin_θ Σ_t ||B_predicted(θ, s_t) - B_observed(t)||²
```

This allows:
- **Personality discovery**: What personality does this existing agent implicitly have?
- **Personality transfer**: Clone a personality from one agent to another
- **Personality fine-tuning**: Start from a base personality and calibrate from user feedback

The `CMAESOptimizer` in `src/meta/` is already designed for this kind of gradient-free optimization over personality parameters.

---

## 9. Implementation roadmap

### Phase 1: Foundation adapters (ScenarioEncoder + ActionEncoder)

**Deliverables:**
- `src/middleware/scenario_encoder.py` — HeuristicEncoder + LLMEncoder + CachedEncoder
- `src/middleware/action_encoder.py` — ActionArchetypeLibrary + classifier
- `src/middleware/types.py` — TaskContext, ActionDescription, ActionResult, ActionGuidance
- Tests: unit tests for each encoder, integration test with existing SDK

**Verification:** Given a free-text task description, produce a valid `Scenario` and `Action` that the existing `DecisionEngine` accepts and processes correctly.

### Phase 2: BehavioralModulator + OutcomeObserver

**Deliverables:**
- `src/middleware/modulator.py` — Temperature, prompt overlay, action guidance
- `src/middleware/outcome_observer.py` — Signal aggregation, personality-modulated weights
- Tests: verify that different personalities produce measurably different `ActionGuidance` for the same task context; verify that outcome observation feeds state transitions correctly

**Verification:** Two agents with different personalities, given the same task, produce different temperatures and system prompt overlays.

### Phase 3: EventDrivenAdapter + BehavioralEvidenceExtractor

**Deliverables:**
- `src/middleware/event_adapter.py` — Lifecycle hooks, idle decay, multi-tick support
- `src/middleware/evidence_extractor.py` — Real behavior → dimension evidence
- Tests: verify that idle decay converges to equilibrium; verify that behavioral evidence from real actions updates psi_hat differently than synthetic evidence

**Verification:** An agent running through a sequence of real tasks accumulates behavioral evidence that causes measurable self-model drift.

### Phase 4: PersonalityMiddleware facade + framework integrations

**Deliverables:**
- `src/middleware/__init__.py` — PersonalityMiddleware facade composing all adapters
- `src/middleware/integrations/` — Reference integrations for Claude Agent SDK, LangChain
- `examples/personality_aware_agent.py` — Working example
- Tests: end-to-end test showing personality affecting agent behavior through the full loop

**Verification:** An end-to-end test where a high-O agent and a high-C agent, given the same coding task, produce measurably different approaches (exploratory vs. systematic), with internal state evolving realistically across the interaction.

---

## 10. Falsification criteria

Following the pattern established in `personality-as-precision-landscape-1.0.md`, this proposal makes testable predictions:

1. **Personality divergence:** Two agents with different personality vectors, given identical task sequences through the middleware, must produce statistically different behavioral traces (action archetype distributions, temperature trajectories, state evolution curves). If personality makes no measurable difference, the middleware is inert.

2. **State coherence:** The temporal state (mood, energy, frustration) must respond realistically to outcome sequences. Repeated failures must increase frustration and decrease energy; repeated successes must increase mood and satisfaction. If state is decoupled from outcomes, the feedback loop is broken.

3. **Self-model accuracy:** After a sequence of real actions, `psi_hat` must converge toward the behavioral evidence profile, not remain fixed at initialization. If the self-model doesn't update, self-evidencing is disconnected.

4. **Emotion functionality:** Constructed emotions must cause measurable behavioral change. An agent in a frustrated state must behave differently (higher escalation rate, lower retry count) than the same agent in a satisfied state. If emotions are cosmetic, the affect engine is decorative.

5. **Temperature grounding:** The precision-derived temperature must produce measurably different LLM output distributions. High-precision states must produce more deterministic outputs; low-precision states must produce more variable outputs. If temperature modulation has no effect, the precision-to-behavior bridge is broken.

---

## 11. Open questions

1. **Encoding stability:** Do the HeuristicEncoder and LLMEncoder produce consistent scenario vectors for semantically equivalent but syntactically different task descriptions? What is the test-retest reliability?

2. **Archetype coverage:** Are 10 action archetypes sufficient, or do real agent behaviors require a larger library? What is the reconstruction error when projecting real actions onto the archetype basis?

3. **Temporal resolution:** What is the right "tick duration" for real agents? Should every LLM call be a tick, or should ticks aggregate over task boundaries?

4. **Multi-personality stability:** When multiple personality-aware agents interact (e.g., a creative agent and a reviewing agent), does the system remain stable, or do personality-driven feedback loops amplify into oscillation?

5. **Personality-performance tradeoff:** Does personality injection improve or degrade task performance? Under what conditions is a personality-aware agent more effective than a personality-agnostic one? The hypothesis is that personality-performance fit (matching personality to task type) improves outcomes, while personality-task mismatch degrades them.

---

## References

- Safron, A. & DeYoung, C. G. (2023). Integrating cybernetic big five theory with the free energy principle. *Springer CCIS*, 1915:73–90.
- Friston, K. (2024). Feeling our way: Active inference, interoception, and self-evidencing. *National Science Review*, 11(5).
- Fisher, S. et al. (2024). Active inference and artificial curiosity. *Entropy*, 26(6):518.
- Laukkonen, R., Friston, K. & Chandaria, S. (2025). The self-evidencing brain and constructed experience. *Neuroscience and Biobehavioral Reviews*.
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3):181–204.
- Friston, K. (2009). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11:127–138.
