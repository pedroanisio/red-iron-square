---
disclaimer: >
  No information in this document should be taken for granted.
  Any statement or premise not backed by a real logical definition
  or verifiable reference may be invalid, erroneous, or a hallucination.
  All mathematical formulas require independent verification against
  the cited primary sources before implementation.
title: "Personality as Precision Landscape: A Hierarchical Self-Evidencing Architecture for Red Iron Square"
version: "1.0"
date: "2026-03-11"
status: "Research architecture proposal with staged migration plan"
---

# Personality as Precision Landscape

## 0. Purpose and scope

This document proposes a single architectural innovation for Red Iron Square: reconceiving personality as **precision allocation across a hierarchical active inference generative model**. Emotions are constructed rather than detected, the LLM serves as the deep generative model within a bounded variational inference loop, and identity emerges from self-evidencing rather than tracking.

The proposal includes a dual-regime computational architecture separating fast math from slow generation, a three-phase staged migration plan grounded in existing Red Iron Square code, behavioral falsification criteria, an ablation protocol, inter-level coupling stability analysis, and a concrete C-vector derivation mapping personality to prior preferences.

This is a research architecture proposal, not a sprint-ready implementation spec. Each migration phase is independently valuable and reversible.

---

## 1. The core thesis

Red Iron Square currently flows linearly:

```
personality vector θ
  → activation functions f_i(s_i, θ_i)
  → utility U = Σ(act_i · mod_i)
  → Boltzmann softmax P(a) = exp(U/τ) / Z
```

The Precision Hyper-Prior (PHP) replaces this with a recursive loop:

```
personality θ parameterizes precision Π
  → Π weights prediction errors ε across hierarchy
  → weighted errors construct emotions
  → emotions update precision
  → precision shapes next prediction
```

This changes the causal role of personality from "input to a utility function" to "the shape of attention, the texture of emotional life, and the rigidity of identity."

### 1.1 Why precision

The core mathematical object in hierarchical predictive coding is the **precision-weighted prediction error**. Precision (π = inverse variance) determines which signals the system amplifies and which it attenuates. In neuroscience, precision maps to synaptic gain.

Safron and DeYoung (2023, Springer CCIS 1915:73–90) formalized the mapping between Big Five traits and active inference parameters. Their result: each personality trait corresponds to a characteristic pattern of precision allocation.

| Trait | Precision role | Current RIS mechanism | PHP mechanism |
|---|---|---|---|
| O (Openness) | Low prior precision → accepts surprise | `f_O = O·tanh(α·s)` magnitude scaling | Epistemic-pragmatic balance in EFE |
| C (Conscientiousness) | High goal-prior precision → stays on task | `f_C = σ(β·(2C-1)·(s-θ))` bipolar sigmoid | Temporal depth, habit strength |
| E (Extraversion) | High policy precision γ → confident action | `f_E = σ(γ·(2E-1)·(s-0.5))` bipolar sigmoid | Tonic policy precision |
| A (Agreeableness) | High precision on social prediction errors | `f_A = A·s + (1-A)·(1-s)` linear interp | Social error weighting |
| N (Neuroticism) | Amplified interoceptive precision | `f_N = exp(-δ·N·s²)` Gaussian decay | Interoceptive precision gain |
| R (Resilience) | Precision recovery rate after perturbation | `f_R = R·(1 - exp(-ρ·s))` mobilization | Precision restabilization speed |
| I (Idealism) | High precision on ideal-outcome priors | `f_I = I·s + (1-I)·(1-s)` linear interp | Goal-prior sharpness |
| T (Tradition) | High precision on self-consistency priors | `f_T = T·s + (1-T)·(1-s)` linear interp | Narrative anchor strength |

### 1.2 On precision universality

A natural concern is that making precision do everything — personality expression, emotion construction, identity rigidity, action confidence — creates an elegant but undebuggable system. This concern is understandable but addressable.

The Free Energy Principle in neuroscience treats precision weighting as a universal mechanism (Friston, 2009; Clark, 2013). This is a theoretical claim about biological cognition, not a proven mathematical property. Whether it holds in the computational setting is an empirical question that this proposal is designed to test.

The debuggability concern is addressed by **hierarchical decomposition**: each level has its own precision variables, prediction errors, and observables, testable independently via the ablation protocol in §8. The inter-level coupling analysis in §6.3 addresses whether the coupled system as a whole remains stable.

---

## 2. The four-level hierarchy

### 2.1 Level 0 — Allostatic (the body budget)

Red Iron Square tracks mood ∈ [-1, 1], arousal ∈ [0, 1], energy ∈ [0, 1], satisfaction ∈ [0, 1], and frustration ∈ [0, 1] as `AgentState`. Under PHP, these become interoceptive variables regulated by allostatic prediction:

$$\varepsilon^{(0)}_i(t) = s_i(t) - \hat{s}_i(\theta, c_t)$$

where $s_i$ ∈ {mood, energy, arousal, satisfaction, frustration} and $\hat{s}_i$ is the predicted allostatic set-point, parameterized by personality θ and context $c_t$.

**Allostatic set-points from personality:**

$$\hat{s}_{\text{arousal}}(\theta) = 0.4 + 0.15 \cdot E$$

$$\hat{s}_{\text{energy}}(\theta) = 0.80$$

$$\hat{s}_{\text{mood}}(\theta) = 0.0$$

$$\hat{s}_{\text{satisfaction}}(\theta) = 0.5$$

$$\hat{s}_{\text{frustration}}(\theta) = 0.0$$

These match the current `StateTransitionParams` defaults (`E_arousal_baseline=0.15`, `energy_resting_level=0.80`, decay centers). The difference: deviations from set-points produce prediction errors, and those errors are precision-weighted before driving state updates.

Note: only arousal has a personality-dependent set-point. For the remaining four variables, personality enters exclusively through precision weighting — a high-N agent has amplified interoceptive precision on mood errors, so the same neutral set-point produces larger weighted prediction errors, functionally similar to a lower baseline mood. Whether this is sufficient or whether additional personality-dependent set-points add expressive power is an empirical question for Phase A shadow-tracking.

**Existing code path:** `temporal/state.py::update_state()` computes transitions with handcrafted equations. The scattered trait-specific coefficients (`N_mood_sensitivity=0.5`, `R_frustration_damping=0.4`) are replaced by a single precision vector applied to prediction errors.

**Reference:** Sennesh, Barrett & Quigley (2022). "Interoception as modeling, allostasis as control." Biological Psychology, 167:108242.

### 2.2 Level 1 — Situated (action selection)

The existing Boltzmann decision engine lives here. Under PHP, utility is replaced by expected free energy (EFE) minimization:

$$G(\pi) = \underbrace{-\mathbb{E}_{q}\bigl[D_{\text{KL}}\bigl(q(s_\tau | o_\tau, \pi) \| q(s_\tau | \pi)\bigr)\bigr]}_{\text{epistemic value (curiosity)}} + \underbrace{\mathbb{E}_{q}\bigl[D_{\text{KL}}\bigl(q(o_\tau | \pi) \| p(o_\tau | C)\bigr)\bigr]}_{\text{pragmatic value (preference)}}$$

Policy selection retains the Boltzmann softmax form:

$$P(\pi) = \sigma(-\gamma \cdot G(\pi))$$

where γ is **policy precision** (how deterministic action selection is). Personality enters through both γ (modulated by Extraversion) and C (the prior preference vector — see §2.5).

**Epistemic value approximation.** The full epistemic value requires A/B matrices and belief-state inference via pymdp, which is a Phase C dependency. For Phase B, epistemic value is approximated as **outcome prediction variance** from the agent's recent memory:

$$V_{\text{epistemic}}(\pi) = \text{Var}\bigl[\{o_t : a_t \in \pi\}\bigr]_{\text{recent}}$$

computed from the existing `MemoryBank` entries filtered by action class. High outcome variance for a given action means the agent is uncertain about its consequences — epistemic value for that action is high. Low variance means the action's effects are well-understood — epistemic value is low. This requires only the stored outcome history already available in `MemoryBank`. Openness modulates the weight of this term in the total EFE:

$$G(\pi) = (1 - w_O) \cdot G_{\text{pragmatic}}(\pi) + w_O \cdot G_{\text{epistemic}}(\pi)$$

where $w_O = O \cdot w_{\text{base}}$ and $w_{\text{base}}$ is a tunable hyperparameter (initial value: 0.3). High-O agents weight epistemic value more, producing more exploration. The full information-gain computation replaces this approximation when pymdp is integrated in Phase C.

**Existing code path:** `DecisionEngine.decide()` computes `logits = utilities / temperature`, `probs = softmax(logits)`, samples via `rng.choice(len(actions), p=probs)`. Under PHP, `utilities` → `−G(π)` and `temperature` → `1/γ`. The softmax form is identical.

**Reference:** Champion, Bowman, Markovic & Grzes (2024). "Reframing the Expected Free Energy." Millidge, Tschantz & Buckley (2021). "Whence the Expected Free Energy?" Neural Computation, 33(2):447–482.

### 2.3 Level 2 — Narrative (self-model)

`self_model/model.py::SelfModel` tracks ψ̂ with:

$$\hat{\psi}_i(t+1) = \hat{\psi}_i(t) + \eta \cdot [B_i(t) - \hat{\psi}_i(t)] - \lambda \cdot [\hat{\psi}_i(t) - \hat{\psi}_{0,i}]$$

where B is behavioral evidence (EMA of probability-weighted action modifiers), η = `learning_rate=0.08`, λ = `identity_inertia=0.04`. Under PHP, this becomes a narrative generative model maintained by the LLM at phase boundaries (not per-tick — see §3). Prediction errors at this level:

$$\varepsilon^{(2)}(t) = \psi(t) - \hat{\psi}(t \mid \text{narrative}_t, \text{memories}_t)$$

The self-model update retains its mathematical form but gains precision weighting: η and λ become functions of personality-derived precision at this level.

**Existing code path:** `SelfModel.update()` and `SelfModel.compute_prediction_error()` already measure self-model surprise. The Level 2 → Level 1 feedback mechanism (self-evidencing) is specified in §5 with stability bounds in §5.1.

**Reference:** Bouizegarene, Ramstead, Constant, Friston & Kirmayer (2024). "Narrative as active inference." Frontiers in Psychology, 15:1345480.

### 2.4 Level 3 — Hyper-model (precision control)

A meta-model takes personality traits, current state, and context to generate precision allocations for all other levels:

$$\Pi_l(t) = \text{softplus}\bigl(W_l \cdot \sigma(\theta) + V_l \cdot s(t) + U_l \cdot c(t) + b_l\bigr)$$

where θ is the 8D personality vector (passed through existing activation functions from `activations.py`), s(t) is the 5D `AgentState` vector, c(t) is a context embedding, and $W_l, V_l, U_l, b_l$ are learned parameters for each level $l$.

**Parameter dimensions.** Define $k$ = precision channels per level, $d$ = context embedding dimension. Parameter count per level: $(8 + 5 + d + 1) \times k$. With the recommended initial configuration of $k = 5$ and $d = 8$:

$$\text{params per level} = (8 + 5 + 8 + 1) \times 5 = 110$$

$$\text{total params} = 110 \times 3 = 330$$

**Total variational free energy across the hierarchy:**

$$F_{\text{total}} = \sum_{l=0}^{2} \sum_i \Pi_{l,i}(t) \cdot \varepsilon_{l,i}^2(t) - \ln \Pi_{l,i}(t)$$

The $-\ln \Pi$ term prevents trivially minimizing free energy by setting all precisions to zero (the Occam factor; Feldman & Friston, 2010, Frontiers in Human Neuroscience).

**Parameter learning strategy.** Three approaches, usable independently or in combination:

1. **Supervised initialization.** Regress against the current handcrafted parameter relationships. For example, `N_mood_sensitivity=0.5` maps to $W_0[\text{mood\_row}, \text{N\_col}]$. This seeds the precision model to reproduce current behavior before any optimization.

2. **Gradient-free optimization via CMA-ES.** The agent minimizes its own $F_{\text{total}}$ inside the simulation loop — this is the active inference decision rule and must not be touched. The meta-optimizer operates *outside* the simulation with a different objective: **minimize behavioral divergence between the simulated agent and a target personality profile.**

$$\mathcal{L}_{\text{meta}} = \sum_{p \in \text{profiles}} D_{\text{KL}}\bigl(\pi_{\text{simulated}}^{(p)} \| \pi_{\text{target}}^{(p)}\bigr) + \lambda_{\text{diversity}} \cdot \mathcal{H}_{\text{between-profile}}$$

where $\pi_{\text{simulated}}^{(p)}$ is the action distribution produced by the agent with personality profile $p$ over a standard test battery, $\pi_{\text{target}}^{(p)}$ is the expected behavioral signature for that profile (derived from the current system's output), and $\mathcal{H}_{\text{between-profile}}$ is a diversity term penalizing personality-collapsed solutions.

**Why the naive objective fails:** If CMA-ES minimizes the agent's own $F_{\text{total}}$ across trajectories, it converges on precision parameters that make every agent perfectly adapted to its environment — zeroing out all prediction errors, erasing personality-dependent behavioral variation. The optimizer tunes personality out of existence. The correct meta-objective preserves personality differentiation by requiring that distinct profiles produce distinct behavioral signatures.

3. **Bayesian calibration.** Place priors on $W, V, U, b$ derived from neuroscience literature (Safron & DeYoung's trait-precision mappings) and update from simulation data. Compatible with methods 1 and 2 as initialization.

### 2.5 The C-vector: deriving prior preferences from personality

The pragmatic value in EFE is $\mathbb{E}_q[D_{\text{KL}}(q(o_\tau|\pi) \| p(o_\tau|C))]$. $C$ encodes prior preferences over observations — what outcomes the agent considers desirable. This is the mechanism through which personality shapes what the agent *wants*, as opposed to how it *attends* (precision) or how it *decides* (policy).

**Observation space.** The agent observes a 5D outcome vector corresponding to changes in its interoceptive state: $o = [\Delta\text{mood}, \Delta\text{energy}, \Delta\text{arousal}, \Delta\text{satisfaction}, \Delta\text{frustration}]$. Each dimension is discretized into $M = 5$ bins: {very negative, negative, neutral, positive, very positive} for mood/satisfaction, and {large decrease, decrease, stable, increase, large increase} for energy/arousal/frustration. The C-vector is factored as a product of independent per-dimension preferences:

$$\ln p(o | C) = \sum_{j=1}^{5} \ln p(o_j | C_j)$$

This reduces $C$ from 3125 entries to five 5-vectors (25 parameters total). The conditional independence assumption is a mean-field approximation: in the actual `update_state()`, mood and frustration interact (negative outcomes amplify mood drop AND increase frustration simultaneously), and energy depends on scenario stress. These cross-dimension correlations mean the factored C-vector may underspecify preferences in crisis states where multiple dimensions move together. Phase B testing should surface this; pairwise interaction terms ($C_{ij}$) can be added if the equivalence test fails on multi-dimension crisis scenarios.

**Derivation from personality.** Each $C_j$ is a categorical distribution over 5 bins for observation dimension $j$:

**Mood preferences $C_{\text{mood}}$:** All agents prefer positive mood changes. Neuroticism and Idealism modulate the distribution.

$$C_{\text{mood}}(k) \propto \exp\bigl(\kappa_{\text{mood}} \cdot (k - k_0) - N \cdot |k - k_0|\bigr)$$

where $k$ indexes the 5 bins (1=very negative, 5=very positive), $k_0 = 3$ (neutral), $\kappa_{\text{mood}} > 0$ is a base preference for positive change. High-N agents have steeper penalty for negative bins (asymmetric aversion). High-I agents shift $\kappa_{\text{mood}}$ upward (stronger preference for ideal outcomes).

**Arousal preferences $C_{\text{arousal}}$:** Extraversion determines preferred arousal level.

$$C_{\text{arousal}}(k) \propto \exp\bigl(-\kappa_{\text{arousal}} \cdot (k - k^*_E)^2\bigr)$$

where $k^*_E = 2 + 2E$ maps Extraversion to a preferred arousal bin (introverts prefer low arousal ~bin 2; extraverts prefer high arousal ~bin 4).

**Energy preferences $C_{\text{energy}}$:** All agents prefer energy stability or increase. Conscientiousness modulates planning horizon preference.

$$C_{\text{energy}}(k) \propto \exp\bigl(\kappa_{\text{energy}} \cdot (k - 3) + C_{\text{trait}} \cdot \mathbf{1}[k = 3]\bigr)$$

High-C agents receive a bonus for the "stable" bin (preference for predictable energy management over volatile gains).

**Satisfaction preferences $C_{\text{satisfaction}}$:** Agreeableness modulates sensitivity to social satisfaction signals.

$$C_{\text{satisfaction}}(k) \propto \exp\bigl((\kappa_{\text{sat}} + A) \cdot (k - 3)\bigr)$$

High-A agents have stronger preference gradients for positive satisfaction.

**Frustration preferences $C_{\text{frustration}}$:** All agents prefer low frustration. Resilience modulates tolerance.

$$C_{\text{frustration}}(k) \propto \exp\bigl(-\kappa_{\text{frust}} \cdot (1 - R) \cdot (k - 1)\bigr)$$

Low-R agents have steep preferences against frustration increase. High-R agents have flatter distributions (they tolerate frustration because they recover faster).

**Trait coverage.** The 8D personality vector maps to $C$ as follows: N and I shape $C_{\text{mood}}$; E shapes $C_{\text{arousal}}$; C shapes $C_{\text{energy}}$; A shapes $C_{\text{satisfaction}}$; R shapes $C_{\text{frustration}}$; O shapes the epistemic-pragmatic balance (the $w_O$ weight in §2.2, not $C$ itself); T shapes self-evidencing precision (§5), not observation preferences.

**Normalization.** Each $C_j$ is normalized to a proper categorical distribution. The $\kappa$ parameters are hyperparameters initialized from the current system's implicit preference structure and tunable via the meta-optimizer (§2.4). Note that the bin-index-based formulas assume linear subjective distance between bins; if this proves too coarse during calibration, the $\kappa$ terms can absorb nonlinear scaling.

**Existing code path bridge.** The current `DecisionEngine` embeds preferences implicitly: `f_A = A·s + (1-A)·(1-s)` means high-A agents respond more strongly to high-cooperation stimuli. Under PHP, this implicit preference becomes the explicit $C_{\text{satisfaction}}$ distribution. The activation function's shape determines the initial $\kappa$ values; the C-vector makes them transparent and independently testable.

---

## 3. The dual-regime computational architecture

The LLM is **not called on every tick**.

### System 1 — Fast math (every tick, <1ms)

All per-tick computation is pure numpy within the existing `TemporalSimulator.tick()` / `SelfAwareSimulator.tick()` pipeline:

- Precision generation: $\Pi_l(t) = \text{softplus}(W_l \cdot \sigma(\theta) + V_l \cdot s(t) + U_l \cdot c(t) + b_l)$ — matrix multiply + softplus, ~0.1ms
- Prediction error: $\varepsilon^{(l)}(t) = \text{observed} - \text{predicted}$ — elementwise subtraction, ~0.01ms
- Precision-weighted free energy: $F_{\text{total}} = \sum \Pi \cdot \varepsilon^2 - \ln \Pi$ — dot product + log, ~0.01ms
- EFE-based action selection: softmax over $-\gamma \cdot G(\pi)$ — identical cost to current Boltzmann
- State update via precision-weighted prediction errors
- Valence from free energy rate of change (§4 Step 3a)
- Self-model update: existing EMA + evidence pull + anchor pull, precision-weighted

Generative model parameters (A-matrix, B-matrix, C-vector, narrative state) are **cached constants** during fast-loop execution.

### System 2 — Slow generation (async, at phase boundaries)

The LLM is invoked at:

1. **Phase boundaries** — campaign phase transitions (via `POST /runs/{run_id}/phases`).
2. **Surprise spikes** — cumulative prediction error at any level exceeds a threshold. Threshold: $\|\tilde{\varepsilon}^{(0)}(t)\| > \mu_\varepsilon + 2\sigma_\varepsilon$ where $\mu$ and $\sigma$ are computed from the last 50 ticks, with a floor $\sigma_{\min} = 0.05$ to prevent trivial triggers during monotonous periods. During the first 50 ticks (warmup), a fixed fallback threshold is used: the existing `coherence_threat_threshold=0.20`.
3. **Assisted steps** — explicitly requested via `POST /runs/{run_id}/assist/step`.
4. **Intervention requests** — via `POST /runs/{run_id}/intervention`.

At these junctures the LLM performs: generative model refresh (A/B matrices), emotion construction (§4 Step 3b), and narrative maintenance (`NarrativeChunk`).

**Latency budget:** System 2 fires approximately 5–15 times per 200-tick simulation, adding 10–75 seconds total. This matches the current assisted-step flow.

---

## 4. Emotion construction, not detection

Red Iron Square detects emotions via heuristic rules in `temporal/affective_engine.py` (e.g., `excitement = max(0, mood) * arousal`, `anxiety = arousal * max(0, -mood) * N`). Under PHP, emotions are **constructed** from precision-weighted prediction error patterns (Barrett, 2017, SCAN, 12(1):1–23).

**Step 1 (System 1, every tick):** Interoceptive prediction errors:

$$\varepsilon^{(0)}(t) = [\Delta\text{mood}, \Delta\text{energy}, \Delta\text{arousal}, \Delta\text{satisfaction}, \Delta\text{frustration}]$$

**Step 2 (System 1, every tick):** Precision-weighted:

$$\tilde{\varepsilon}^{(0)}_i(t) = \Pi_{0,i}(t) \cdot \varepsilon^{(0)}_i(t)$$

**Step 3a (System 1, every tick) — Valence from free energy rate of change:**

Hesp et al. (2021, Neural Computation, 33(2):398–446) define valence as the rate of change of expected free energy. Since System 1 already computes $F_{\text{total}}$ on every tick:

$$\text{valence}(t) = F_{\text{total}}(t-1) - F_{\text{total}}(t)$$

Positive valence = free energy is decreasing (things are going better than predicted). Negative valence = free energy is increasing (things are going worse). This is a single scalar subtraction of a quantity already computed — zero additional cost.

**Arousal signal:**

$$\text{arousal\_signal}(t) = \|\tilde{\varepsilon}^{(0)}(t)\|$$

This gives continuous affect coordinates on every tick without LLM involvement.

**Step 3b (System 2, at surprise spikes):** When $\|\tilde{\varepsilon}^{(0)}(t)\|$ exceeds the threshold (§3), the LLM categorizes the error pattern into a discrete emotion concept given context:

$$\text{emotion}(t) = \text{LLM}_{\text{categorize}}\bigl(\tilde{\varepsilon}^{(0)}(t), \text{context}(t), \text{memory}(t)\bigr)$$

The same prediction error pattern can be categorized as "anxiety" in one context and "excitement" in another — exactly as Barrett's theory proposes. To prevent narratively plausible but psychologically inconsistent categorizations, the `EmotionConstructor` schema includes valence/arousal constraints derived from the System 1 signal. The LLM-constructed emotion must be consistent with the sign of valence and the magnitude of arousal from Step 3a, or it is rejected and the system falls back to the nearest heuristic label.

**Mood as slow hyperprior (Hesp et al., 2021):**

$$\text{mood}(t+1) = \alpha \cdot \text{mood}(t) + (1 - \alpha) \cdot \text{valence}(t)$$

where α is personality-dependent: high R → faster recovery → smaller α.

**Personality shapes emotion construction through precision.** A high-N agent has amplified interoceptive precision ($\Pi_{0,i}$ large), so small deviations produce large weighted errors, leading to more frequent surprise spikes and more emotion construction events. A high-O agent has more flexible emotion categories (weaker priors on which concept applies), allowing novel emotional experiences.

---

## 5. Self-evidencing identity

The existing self-model tracks ψ̂ divergence from true ψ (self-accuracy) and from behavioral evidence B (coherence gap). Under PHP, the agent actively **self-evidences**: it selects actions that confirm its generative model's predictions about itself.

**Mechanism:** At Level 2, the self-model generates predictions about what kind of agent this is and what it will do. These predictions constrain policy selection at Level 1 through precision weighting:

$$\Pi_{1,\pi}^{\text{self}}(t) = \Pi_1^{\text{base}}(t) \cdot \exp\bigl(-\beta \cdot d(\pi, \hat{\psi}(t))\bigr)$$

where $d(\pi, \hat{\psi})$ measures divergence between a candidate policy and the self-model's predictions, and β is personality-dependent: high T → high β → strong self-consistency; high O → low β → flexible identity.

**Existing code path:** `SelfModel.predict_action_distribution()` already computes predicted action probabilities from ψ̂. `SelfModel.compute_prediction_error()` measures cross-entropy between predicted and actual distributions. The self-evidencing precision term adds a new feedback path from Level 2 prediction to Level 1 policy precision.

Identity drift (`current_identity_drift()`) is reinterpreted as cumulative narrative prediction error. Self-coherence (`current_coherence_gap()`) is the instantaneous gap between behavior and self-concept. The self-emotions — pride, shame, authenticity, identity threat, identity crisis — already detected in `self_model/emotions.py` — become precision-weighted signals.

**Reference:** Friston (2024). "Self-evidencing." National Science Review, 11(5). Fisher et al. (2024). "Universal optimism of the self-evidencing mind." Entropy, 26(6):518. Laukkonen, Friston & Chandaria (2025). "A beautiful loop." Neuroscience and Biobehavioral Reviews, 176:106296.

### 5.1 Stability bounds for self-evidencing

The self-evidencing term creates a positive feedback loop: consistent behavior → stronger self-model → higher precision on consistent actions → more consistent behavior. Without bounds, this loop can degenerate into a single-action attractor. Three stabilization mechanisms prevent this:

**Mechanism A — Precision cap:**

$$\Pi_{1,\pi}^{\text{self}}(t) = \Pi_1^{\text{base}}(t) \cdot \min\bigl(\Pi_{\max}, \exp(-\beta \cdot d(\pi, \hat{\psi}(t)))\bigr)$$

where $\Pi_{\max} = 3.0$ (the self-model can at most triple a policy's base precision).

**Mechanism B — Temporal decay:**

$$\beta(t+1) = \beta_{\min} + (\beta(t) - \beta_{\min}) \cdot \lambda_{\beta}$$

where $\lambda_{\beta} = 0.95$ (slow decay per tick), $\beta_{\min} = T \cdot \beta_0$ (personality-dependent floor). Between LLM narrative updates, self-evidencing gradually loosens. Each System 2 narrative refresh can reset β based on updated self-model confidence.

**Mechanism C — Policy-normalized precision:**

$$\Pi_{1,\pi}^{\text{self-norm}}(t) = \Pi_1^{\text{base}}(t) \cdot \frac{\exp(-\beta \cdot d(\pi, \hat{\psi}))}{\frac{1}{|\Pi|}\sum_{\pi'} \exp(-\beta \cdot d(\pi', \hat{\psi}))}$$

This conserves total precision budget: if one policy gains precision, others lose it.

**Recommended approach:** Use mechanisms A and C together. The cap prevents extreme values; the normalization preserves the precision budget. Mechanism B provides additional safety for very long runs (>1000 ticks). The ablation protocol (§8) tests each mechanism independently — note that the cap-removal ablation is a **two-sided test**: if removing the cap does NOT produce degenerate attractors, either the cap is unnecessary overhead or the test scenarios lack sufficient pathological coverage.

---

## 6. Behavioral falsification and stability

### 6.1 Predicted behavioral signatures

**Prediction 1 — Context-dependent emotion construction.** Identical agent state → different emotion labels in different narrative contexts. The current `AffectiveEngine` produces identical readings for identical state. Test: two simulations, identical state trajectories, different scenario narratives; compare emotion label distributions.

**Prediction 2 — O/C exploration-exploitation tradeoff.** High-O agents explore more (higher epistemic value weighting $w_O$ in EFE), high-C agents exploit more (higher pragmatic value). Test: action entropy across 1000-tick runs for high-O vs. high-C profiles; PHP should show statistically significant difference.

**Prediction 3 — Self-evidencing behavioral attractors.** High-T agents converge to narrower behavioral repertoires as self-consistency precision strengthens. Test: action distribution entropy over time for high-T vs. low-T; PHP should show diverging entropy curves.

**Prediction 4 — Nonlinear N-stress response.** Under stress, high-N agents show disproportionate disruption. PHP predicts a threshold effect (sigmoidal divergence). Test: sweep scenario-N from 0 to 1, measure mood variance; PHP should show sigmoidal pattern, current system shows linear.

**Prediction 5 — Narrative coherence recovery.** After disruption, PHP agents show narrative repair (non-exponential recovery shape). Test: inject bad outcomes, return to neutral, measure coherence recovery shape.

### 6.2 Falsification conditions

The PHP architecture is falsified if:

1. Precision weighting produces no measurable behavioral difference vs. current linear activation model across diverse personality profiles.
2. LLM-based emotion construction produces lower inter-rater agreement on plausibility than heuristic detection.
3. Learned precision parameters converge to values effectively equivalent to current handcrafted coefficients (no added expressive power).
4. Self-evidencing feedback creates degenerate single-action loops despite stability mechanisms in §5.1.

### 6.3 Inter-level coupling stability analysis

The PHP architecture has three levels whose precision allocations interact in a coupled dynamical system:

```
Level 0 (allostatic errors) → feed into Level 1 (action selection via EFE)
Level 1 (action outcomes) → feed into Level 2 (narrative self-model update)
Level 2 (narrative precision) → feeds back into Level 1 (self-evidencing boost)
```

This creates a closed loop: L0 → L1 → L2 → L1 → (through action) → L0.

**Analysis by coupling direction:**

**L0 → L1 (allostatic → action).** Interoceptive prediction errors modulate the EFE pragmatic term through the C-vector: large allostatic errors bias the agent toward corrective actions. This is a negative feedback loop (error → corrective action → reduced error) and is inherently stabilizing.

**L1 → L2 (action → narrative).** Action outcomes update behavioral evidence B via EMA (`evidence_memory=0.85`). The EMA acts as a low-pass filter: rapid action changes are smoothed, preventing Level 2 from oscillating. The time constant (0.85 per tick) means Level 2 responds on a ~7-tick timescale, much slower than Level 1's per-tick decisions. This temporal separation is stabilizing.

**L2 → L1 (narrative → action, self-evidencing).** The only potentially destabilizing coupling. The stability mechanisms in §5.1 bound the gain: with $\Pi_{\max} = 3.0$ and policy normalization, the self-evidencing term can concentrate precision but cannot create precision from nothing.

**Overall stability argument.** The L0 → L1 → L0 sub-loop is a standard negative feedback (homeostatic) system. The L2 → L1 positive feedback is bounded by §5.1 and filtered by the EMA time constant in L1 → L2. Systems with well-separated timescales are generally stable when the fast loop is negative-feedback and the slow loop's gain is bounded (Khalil, 2002, "Nonlinear Systems," Ch. 11: singular perturbations).

**Pathological configurations.** An agent with simultaneously high N (amplified L0 errors), low R (slow precision recovery), and high T (strong L2 → L1 coupling) could enter an oscillatory regime where allostatic distress amplifies self-inconsistency, which tightens self-evidencing, which restricts action repertoire, which prevents recovery. This is computationally analogous to an anxiety-rigidity spiral. Whether this is a bug or a feature depends on whether such spirals are psychologically realistic.

**Pre-implementation gating test.** Before Phase C, run a parameter sweep across all 256 corner configurations of the 8D personality space (each trait at 0 or 1) and simulate 1000 ticks each. Flag any configuration where: (a) action entropy drops below 0.1 nats (degenerate attractor), (b) mood oscillates with period <10 ticks and amplitude >0.5 (instability), or (c) free energy diverges. This sweep costs ~0.3ms/tick × 1000 ticks × 256 configs ≈ 77 seconds total.

---

## 7. Staged migration plan

Each phase is independently valuable and reversible.

### 7.1 Phase A — Precision as explicit tracked state

**What changes:** Add precision variables alongside the existing activation/utility/Boltzmann pipeline. Precision is computed, tracked, and persisted — but does not yet influence decisions.

**Concrete code changes:**

1. Add `PrecisionState` model:
   ```python
   class PrecisionState(BaseModel):
       level_0: np.ndarray  # 5D: one per interoceptive variable
       level_1: float       # policy precision γ
       level_2: float       # narrative precision
   ```

2. Add `PrecisionEngine` that computes Π from personality + state using the softplus equation.

3. Compute prediction errors at Level 0 (allostatic set-point deviations) as observables. Add to `TickResult`.

4. Ablation test: compare `PrecisionState` trajectories across personality profiles.

**What does not change:** `DecisionEngine`, `update_state()`, `AffectiveEngine`, `SelfModel`.

**Exit criterion:** Precision trajectories show expected personality-dependent patterns (high-N → high interoceptive precision, high-E → high policy precision) across a standard test battery of at least 8 personality profiles (each extreme of each trait pair) with 10 seeded runs each.

### 7.2 Phase B — Swap utility for lightweight EFE surrogate

**What changes:** Replace the utility dot-product with an EFE approximation decomposed into epistemic and pragmatic components.

**Concrete code changes:**

1. Add `EFEEngine` implementing the same interface as `DecisionEngine`:
   ```python
   class EFEEngine:
       def __init__(self, C_vector: CVector, ...):
           self._C = C_vector  # derived from personality per §2.5

       def utility(self, personality, scenario, action, ...) -> float:
           pragmatic = self._pragmatic_value(action, self._C)
           epistemic = self._epistemic_value(action, memory_stats)
           return -(pragmatic + w_O * epistemic)

       def decide(self, ...) -> tuple[Action, np.ndarray]:
           # Same Boltzmann softmax, temperature = 1/γ
           # γ = policy precision from PrecisionState
   ```

2. `CVector` constructed from personality per §2.5 (5 categorical distributions × 5 bins = 25 parameters).

3. Epistemic value computed as outcome variance from `MemoryBank` per §2.2.

4. `AgentSDK` gains a constructor parameter to select `DecisionEngine` vs. `EFEEngine`.

**What does not change:** `update_state()`, `AffectiveEngine`, `SelfModel`.

**Exit criteria:**

- **Equivalence test:** For balanced personality profiles (all traits at 0.5), action distribution KL-divergence between `DecisionEngine` and `EFEEngine` ≤ 0.1 nats, measured over 1000 ticks with 10 seeded runs.
- **Differentiation test:** For extreme O/C profiles (O=0.9/C=0.1 vs. O=0.1/C=0.9), action entropy difference ≥ 0.3 nats (high-O more entropic than high-C), measured over 1000 ticks with 10 seeds.
- Both must pass simultaneously. Equivalence pass + differentiation fail → C-vector recalibration needed. Equivalence fail → EFE surrogate misconfigured.

### 7.3 Phase C — Emotion construction and narrative generative model

Phase C combines three novel subsystems. To manage integration risk, it is split into two sub-phases:

**Phase C1 — Constructed emotion (no self-evidencing).** Tests Predictions 1 and 4.

1. `ConstructedAffectiveEngine` (alternative to `AffectiveEngine`):
   - System 1 (every tick): valence from $\Delta F$ (§4 Step 3a), arousal from $\|\tilde{\varepsilon}^{(0)}\|$.
   - System 2 (on surprise spike): LLM emotion categorization via `EmotionConstructor` schema with valence/arousal consistency constraints.

2. `NarrativeGenerativeModel` with cached A/B/C matrices:
   ```python
   class NarrativeGenerativeModel:
       def __init__(self, adapter: StructuredLLMAdapter): ...
       def refresh(self, trajectory_window, goals) -> GenerativeModelParams: ...
       def cached_A(self) -> np.ndarray: ...
       def cached_B(self) -> np.ndarray: ...
       def cached_C(self) -> CVector: ...
   ```

**Phase C2 — Add self-evidencing feedback.** Tests Predictions 3 and 5. Level 2 prediction error feeds into Level 1 policy precision per §5, with stability bounds per §5.1. Gated by the 256-config stability sweep (§6.3).

**What does not change:** `StructuredLLMAdapter` protocol, `AgentRuntime` interface, the principle that LLM outputs are typed, validated, and never mutate state directly.

### 7.4 Phase C integration risks

**Emotion-precision coupling.** Constructed emotions update mood (via valence), which changes `AgentState`, which changes precision generation (via $V_l \cdot s(t)$), which changes prediction errors, which changes constructed emotions. This closed loop is stabilized by the mood EMA ($\alpha \approx 0.9$) acting as a low-pass filter, and precision generation bounded by softplus monotonicity. But interaction between this loop and the L0→L1→L2 loop is untested until C1+C2 are combined.

**LLM cache staleness.** Between System 2 refreshes, the cached generative model may become inaccurate as agent state drifts. The adaptive surprise-spike threshold (§3) controls refresh frequency. If the threshold is too high, parameters go stale. If too low, the system approaches per-tick LLM invocation. The $2\sigma$ threshold with $\sigma_{\min} = 0.05$ floor is a starting point; empirical tuning is expected.

**Constructive emotion hallucination.** The LLM may categorize prediction error patterns into emotion concepts that are narratively plausible but psychologically inconsistent. Mitigation: the `EmotionConstructor` schema includes valence/arousal constraints from System 1. Inconsistent categorizations are rejected; fallback is the nearest heuristic label.

---

## 8. Ablation protocol

| Ablation | What's removed | Expected effect | Confirms |
|---|---|---|---|
| No precision weighting | All Π = 1 (uniform) | Collapses to current RIS behavior | Precision adds signal |
| No epistemic value | Remove epistemic term from EFE | Reduced exploration for high-O | O-precision mapping works |
| No self-evidencing | Remove L2 → L1 feedback | No attractor narrowing for high-T | T-precision mapping works |
| No emotion construction | Use heuristic AffectiveEngine | No context-dependent emotion | Construction adds value |
| No allostatic set-points | Use current update_state() | No nonlinear N-stress | Allostatic framing adds value |
| Learned vs fixed Π | Compare learned W,V,U,b vs hand-tuned | Learned shows lower $\mathcal{L}_{\text{meta}}$ | Learning worth complexity |
| Self-evidencing cap removed | Remove §5.1 mechanisms | Two-sided: degenerate attractors confirm necessity; no degeneracy suggests self-stabilization or insufficient test coverage | Stability bounds necessary or not |

---

## 9. Migration cost assessment

| Component | Current mechanism | PHP mechanism | Distance |
|---|---|---|---|
| Action selection | Utility dot-product + Boltzmann | EFE minimization + Boltzmann | **Medium** |
| State transitions | Handcrafted `update_state()` | Allostatic prediction error minimization | **High** |
| Emotion detection | Heuristic `AffectiveEngine` | Precision-weighted construction + LLM | **High** |
| Self-model update | EMA evidence + anchor pull | Precision-weighted evidence + narrative | **Medium** |
| LLM integration | Typed advisor at explicit junctures | Narrative generative model at surprise spikes | **Low** |
| Personality role | Static vector → activations → utility | Dynamic precision allocator → shapes all | **High** |

Estimated effort: Phase A ~2 weeks, Phase B ~3 weeks, Phase C1+C2 ~4–7 weeks. Total: 6–12 weeks. Each phase independently valuable.

---

## 10. pymdp integration path

1. **Constrained structured output.** Existing `StructuredLLMAdapter.complete_json()` validates against Pydantic models. Add `MatrixProposal` with shape, non-negativity, and normalization constraints.

2. **Small state/action spaces.** 2–5 actions, 5 interoceptive variables discretized into 5 bins. A-matrices at most 5×5×5, B-matrices 5×5×5.

3. **Fallback.** If LLM fails after N retries, fall back to matrices derived from current activation functions. System degrades to current behavior, never crashes.

4. **Incremental adoption.** Phase B does not require pymdp. Full integration is a Phase C enhancement, replacing the memory-variance epistemic approximation (§2.2) with proper information-gain computation.

**Reference:** Heins, Millidge, Da Costa, et al. (2022). "pymdp: A Python library for active inference in discrete state spaces." JOSS, 7(73):4098.

---

## 11. References

1. Friston, K. (2009). "The free-energy principle: a rough guide to the brain?" Trends in Cognitive Sciences, 13(7):293–301.
2. Friston, K. (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience, 11:127–138.
3. Feldman, H. & Friston, K. (2010). "Attention, uncertainty, and free-energy." Frontiers in Human Neuroscience, 4:215.
4. Clark, A. (2013). "Whatever next? Predictive brains, situated agents, and the future of cognitive science." Behavioral and Brain Sciences, 36(3):181–204.
5. Barrett, L.F. (2017). "The theory of constructed emotion." SCAN, 12(1):1–23.
6. Hesp, C., Smith, R., Parr, T., Allen, M., Friston, K.J. & Ramstead, M.J.D. (2021). "Deeply Felt Affect." Neural Computation, 33(2):398–446.
7. Millidge, B., Tschantz, A. & Buckley, C.L. (2021). "Whence the Expected Free Energy?" Neural Computation, 33(2):447–482.
8. Sennesh, E., Barrett, L.F. & Quigley, K.S. (2022). "Interoception as modeling, allostasis as control." Biological Psychology, 167:108242.
9. Khalil, H.K. (2002). Nonlinear Systems. 3rd ed. Prentice Hall.
10. Safron, A. & DeYoung, C.G. (2023). "Integrating Cybernetic Big Five Theory with the Free Energy Principle." Springer CCIS, 1915:73–90.
11. Heins, C., Millidge, B., Da Costa, L., et al. (2022). "pymdp." JOSS, 7(73):4098.
12. Bouizegarene, N., Ramstead, M.J.D., Constant, A., Friston, K.J. & Kirmayer, L.J. (2024). "Narrative as active inference." Frontiers in Psychology, 15:1345480.
13. Champion, T., Bowman, H., Markovic, D. & Grzes, M. (2024). "Reframing the Expected Free Energy."
14. Laukkonen, R.E., Friston, K.J. & Chandaria, S. (2025). "A beautiful loop." Neuroscience and Biobehavioral Reviews, 176:106296.
15. Friston, K.J. (2024). "Self-evidencing." National Science Review, 11(5).
16. Fisher, S.R., et al. (2024). "Universal optimism of the self-evidencing mind." Entropy, 26(6):518.
17. Wen, Z. (2025). "The Missing Reward." arXiv:2508.05619.
18. Cox, M.T., et al. (2016). "MIDCA: A Metacognitive, Integrated Dual-Cycle Architecture."
19. Petter, S., et al. (2025). "Emotion regulation, fast or slow." Emotion.
20. Sutton, R.S., Precup, D. & Singh, S. (1999). "Between MDPs and semi-MDPs."
