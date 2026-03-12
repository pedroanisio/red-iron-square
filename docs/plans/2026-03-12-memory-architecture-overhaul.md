# Memory Architecture Overhaul (L5) — Implementation Plan

> **Execution note:** Use the local `executing-plans` skill to implement this plan task-by-task.

**Goal:** Replace the flat `MemoryBank` deque with a multi-system memory architecture (episodic, semantic, procedural, working, prospective, autobiographical) with consolidation, forgetting curves, and causal tagging — while preserving all existing query interfaces and integration points.

**Architecture:** Introduce a `MemorySystem` facade that composes six specialized stores behind a single interface. The existing `MemoryBank` becomes the episodic layer (enhanced with forgetting and causal links). New layers are opt-in: callers that only need episodic queries continue working unchanged. The consolidation cycle runs offline after configurable N ticks.

**Tech Stack:** Python 3.11+, Pydantic, numpy, structlog, pytest, ruff, mypy

**LOC Budget:**

| New/Modified File | Estimated LOC | Notes |
|---|---|---|
| `src/memory/__init__.py` | ~5 | Package marker |
| `src/memory/episodic.py` | ~180 | Enhanced MemoryBank + forgetting + causal tags |
| `src/memory/semantic.py` | ~150 | Pattern extraction from episodic clusters |
| `src/memory/procedural.py` | ~120 | Action-context success rate tracking |
| `src/memory/working.py` | ~100 | Capacity-limited active buffer |
| `src/memory/prospective.py` | ~120 | Goal tracking with deadline/priority |
| `src/memory/autobiographical.py` | ~140 | Key events + narrative identity |
| `src/memory/consolidation.py` | ~180 | Offline consolidation cycle |
| `src/memory/system.py` | ~200 | MemorySystem facade composing all layers |
| `src/memory/params.py` | ~80 | Hyperparameters for all memory subsystems |
| `src/temporal/memory.py` | ~120 | Backward-compat shim (delegates to MemorySystem) |
| `src/temporal/simulator.py` | ~10 delta | Wire consolidation trigger |
| `src/efe/epistemic.py` | ~15 delta | Use procedural memory when available |
| `src/self_model/model.py` | ~10 delta | Accept autobiographical events |
| `src/temporal/system2.py` | ~15 delta | Enrich trajectory window with semantic context |
| `tests/test_memory_episodic.py` | ~250 | Forgetting, causal tags, retention |
| `tests/test_memory_semantic.py` | ~200 | Pattern extraction, rule queries |
| `tests/test_memory_procedural.py` | ~180 | Action stats, success rates |
| `tests/test_memory_working.py` | ~150 | Capacity limits, priority eviction |
| `tests/test_memory_prospective.py` | ~150 | Goal lifecycle, deadline tracking |
| `tests/test_memory_autobiographical.py` | ~150 | Key events, narrative queries |
| `tests/test_memory_consolidation.py` | ~200 | Full consolidation cycle |
| `tests/test_memory_system.py` | ~250 | Facade integration, backward compat |

**Total new source:** ~1,275 LOC across 11 files (avg ~116 LOC/file, all under 300 LOC limit)
**Total new tests:** ~1,530 LOC across 8 files

---

## Dependency Graph

```
Task 1: params.py (no deps)
Task 2: episodic.py (depends on: params)
Task 3: procedural.py (depends on: params)
Task 4: working.py (depends on: params)
Task 5: semantic.py (depends on: params, episodic)
Task 6: prospective.py (depends on: params)
Task 7: autobiographical.py (depends on: params, episodic)
Task 8: consolidation.py (depends on: episodic, semantic, procedural)
Task 9: system.py facade (depends on: all memory layers)
Task 10: backward-compat migration (depends on: system facade)
Task 11: integration wiring (depends on: migration)
```

---

## Task 1: Memory Hyperparameters

**File:** `src/memory/params.py` (~80 LOC)
**Tests:** Included in `tests/test_memory_episodic.py` (param validation subset)

**What:** Central Pydantic config for all memory subsystem parameters.

```python
class MemoryParams(BaseModel):
    """Hyperparameters for the multi-system memory architecture."""

    # Episodic
    episodic_max_size: int = 500
    forgetting_lambda: float = 0.02          # Ebbinghaus decay rate
    forgetting_initial_retention: float = 1.0
    emotional_significance_threshold: float = 0.5  # |valence| above this = significant
    causal_lookback: int = 3                  # ticks to scan for cause

    # Semantic
    semantic_max_rules: int = 100
    pattern_min_support: int = 5             # min episodes to extract a rule
    pattern_confidence_threshold: float = 0.7

    # Procedural
    procedural_ema_alpha: float = 0.1        # success rate EMA decay
    procedural_min_samples: int = 3          # min before trusting stats

    # Working memory
    working_capacity: int = 5                # Miller's 7±2, conservative
    working_decay_rate: float = 0.1          # relevance decay per tick

    # Prospective
    goal_max_active: int = 10
    goal_default_priority: float = 0.5

    # Autobiographical
    autobio_max_events: int = 200
    autobio_significance_threshold: float = 0.6

    # Consolidation
    consolidation_interval: int = 25         # ticks between consolidation runs
    consolidation_retention_floor: float = 0.1  # minimum retention before pruning
```

**Acceptance criteria:**
- [ ] All fields have sensible defaults
- [ ] Pydantic validation prevents negative values for rates/sizes
- [ ] mypy strict passes
- [ ] Docstring with field descriptions

---

## Task 2: Enhanced Episodic Memory

**File:** `src/memory/episodic.py` (~180 LOC)
**Tests:** `tests/test_memory_episodic.py` (~250 LOC)

**What:** Migrate and enhance the existing `MemoryBank` with three new capabilities:
1. **Ebbinghaus forgetting curve** — each entry has a `retention` score that decays over ticks
2. **Causal tagging** — link actions to outcomes via `causal_context` field
3. **Emotional significance** — flag high-|valence| entries for consolidation priority

### Data model

```python
class EpisodicEntry(BaseModel):
    """Enhanced episodic memory entry with forgetting and causal context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick: int
    scenario_name: str
    action_name: str
    outcome: float
    counterfactual: float
    state_snapshot: AgentState
    valence: float

    # New fields (L5 enhancements)
    retention: float = 1.0                # Ebbinghaus: decays over ticks
    emotional_significance: float = 0.0    # |valence| at storage time
    causal_context: str = ""               # "action_X in scenario_Y -> outcome"
    cluster_id: int | None = None          # set by semantic consolidation
```

### Core class

```python
class EpisodicMemory:
    """Episodic memory with forgetting curves and causal tagging.

    Backward-compatible: exposes the same query interface as MemoryBank
    plus forgetting-aware retrieval and significance filtering.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    # --- Storage ---
    def store(self, entry: EpisodicEntry) -> None: ...

    # --- Existing queries (backward-compat) ---
    def recent(self, n: int) -> list[EpisodicEntry]: ...
    def mean_outcome(self, window: int = 10) -> float: ...
    def mean_valence(self, window: int = 10) -> float: ...
    def mean_arousal(self, window: int = 10) -> float: ...
    def peak_valence(self, window: int = 50) -> float: ...
    def total_regret(self, window: int = 10) -> float: ...
    def consecutive_failures(self) -> int: ...
    def outcome_variance(self, window: int = 10) -> float: ...
    def action_outcome_variance(self, action_name: str, window: int = 50) -> float | None: ...

    # --- New queries (L5) ---
    def retained(self, current_tick: int, min_retention: float = 0.1) -> list[EpisodicEntry]:
        """Return entries whose retention > min_retention at current_tick."""

    def significant(self, threshold: float | None = None) -> list[EpisodicEntry]:
        """Return emotionally significant entries."""

    def by_scenario(self, scenario_name: str, n: int = 50) -> list[EpisodicEntry]:
        """Return entries for a specific scenario, most recent first."""

    def decay_all(self, current_tick: int) -> int:
        """Apply Ebbinghaus forgetting: retention *= exp(-lambda * dt).
        Returns number of entries pruned below floor."""

    def prune_forgotten(self, floor: float) -> int:
        """Remove entries with retention below floor. Returns count pruned."""
```

### Forgetting curve

```
retention(t) = initial_retention * exp(-lambda * (current_tick - entry.tick))
```

Applied lazily on `retained()` queries and eagerly during consolidation via `decay_all()`.

### Causal tagging

On `store()`, auto-generate `causal_context` from the entry fields:
```python
causal_context = f"{action_name}@{scenario_name}->{'+'if outcome>0 else '-'}{abs(outcome):.2f}"
```

This string is consumed by semantic extraction (Task 5) to identify action-outcome patterns.

**Acceptance criteria:**
- [ ] All 8 existing `MemoryBank` query methods produce identical results
- [ ] `decay_all()` correctly applies exponential forgetting
- [ ] `prune_forgotten()` removes entries below floor
- [ ] `significant()` filters by `emotional_significance`
- [ ] `retained()` filters by computed retention at given tick
- [ ] `causal_context` auto-populated on store
- [ ] Test: forgetting curve reduces retention over 100 ticks
- [ ] Test: significant memories survive pruning longer
- [ ] Test: backward-compat — all existing MemoryBank tests pass against EpisodicMemory

---

## Task 3: Procedural Memory

**File:** `src/memory/procedural.py` (~120 LOC)
**Tests:** `tests/test_memory_procedural.py` (~180 LOC)

**What:** Track action-context success rates as exponential moving averages. Feeds into EFE epistemic value computation with better signal than raw outcome variance.

### Data model

```python
class ActionRecord(BaseModel):
    """Running statistics for an action in a specific context."""

    action_name: str
    context_key: str         # scenario_name or scenario cluster
    attempts: int = 0
    success_rate: float = 0.5  # EMA of positive outcomes
    mean_outcome: float = 0.0  # EMA of raw outcomes
    last_tick: int = 0
```

### Core class

```python
class ProceduralMemory:
    """Tracks action-context success rates via exponential moving averages.

    Learns which actions work in which contexts, providing
    richer signal than raw episodic variance for exploration decisions.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def record(self, action_name: str, context_key: str, outcome: float, tick: int) -> None:
        """Update success rate EMA for the given action-context pair."""

    def success_rate(self, action_name: str, context_key: str) -> float | None:
        """Return success rate for action in context. None if insufficient data."""

    def mean_outcome(self, action_name: str, context_key: str) -> float | None:
        """Return mean outcome EMA for action in context."""

    def best_action(self, context_key: str) -> str | None:
        """Return action with highest success rate in this context."""

    def action_stats(self, action_name: str) -> list[ActionRecord]:
        """All context records for an action (cross-context comparison)."""

    def all_records(self) -> list[ActionRecord]:
        """All tracked action-context pairs."""
```

### EMA update

```python
alpha = params.procedural_ema_alpha
record.mean_outcome = alpha * outcome + (1 - alpha) * record.mean_outcome
record.success_rate = alpha * (1.0 if outcome > 0 else 0.0) + (1 - alpha) * record.success_rate
record.attempts += 1
record.last_tick = tick
```

**Acceptance criteria:**
- [ ] EMA converges to true success rate over 100+ records
- [ ] `success_rate()` returns None when `attempts < min_samples`
- [ ] `best_action()` returns highest success rate action
- [ ] Test: action that succeeds 80% of the time converges to ~0.8 success rate
- [ ] Test: multiple contexts tracked independently

---

## Task 4: Working Memory

**File:** `src/memory/working.py` (~100 LOC)
**Tests:** `tests/test_memory_working.py` (~150 LOC)

**What:** A capacity-limited active buffer (default 5 items) that tracks what the agent is currently attending to. Items have priority and decay — lowest-priority items are evicted when capacity is reached.

### Data model

```python
class WorkingMemoryItem(BaseModel):
    """An item in the capacity-limited working memory buffer."""

    key: str                  # unique identifier
    content: str              # human-readable description
    source: str               # "episodic", "goal", "scenario", "emotion"
    priority: float = 0.5     # [0, 1] — higher = more attended
    inserted_tick: int = 0
    last_accessed_tick: int = 0
```

### Core class

```python
class WorkingMemory:
    """Capacity-limited active buffer modeling attentional focus.

    Evicts the lowest-priority item when capacity is exceeded.
    Priority decays each tick to model fading attention.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def insert(self, item: WorkingMemoryItem) -> WorkingMemoryItem | None:
        """Insert item, evicting lowest-priority if at capacity.
        Returns evicted item or None."""

    def access(self, key: str, tick: int) -> WorkingMemoryItem | None:
        """Access item by key, refreshing its last_accessed_tick."""

    def contains(self, key: str) -> bool: ...

    def current_items(self) -> list[WorkingMemoryItem]:
        """All items sorted by priority descending."""

    def decay_priorities(self, current_tick: int) -> None:
        """Reduce priority of all items based on ticks since last access."""

    def clear(self) -> None: ...

    def __len__(self) -> int: ...
```

### Priority decay

```python
for item in items:
    dt = current_tick - item.last_accessed_tick
    item.priority *= (1.0 - params.working_decay_rate) ** dt
```

**Acceptance criteria:**
- [ ] Cannot exceed capacity (default 5)
- [ ] Lowest-priority item evicted on overflow
- [ ] `decay_priorities()` reduces all priorities
- [ ] `access()` refreshes last_accessed_tick
- [ ] Test: insert 7 items into capacity-5 buffer, verify 2 evictions
- [ ] Test: priority decay over 10 ticks

---

## Task 5: Semantic Memory

**File:** `src/memory/semantic.py` (~150 LOC)
**Tests:** `tests/test_memory_semantic.py` (~200 LOC)

**What:** Extract abstract rules and patterns from episodic memory clusters. Rules are action-context-outcome generalizations (e.g., "bold actions in high-stress scenarios tend toward negative outcomes").

### Data model

```python
class SemanticRule(BaseModel):
    """An extracted pattern from episodic memory clusters."""

    rule_id: str                # auto-generated hash
    pattern: str                # human-readable: "bold@high_stress -> negative"
    action_pattern: str         # action name or wildcard
    context_pattern: str        # scenario name or attribute predicate
    outcome_direction: float    # mean outcome across supporting episodes
    confidence: float           # support_count / total_relevant * correlation
    support_count: int          # number of supporting episodes
    extracted_tick: int         # when this rule was created
```

### Core class

```python
class SemanticMemory:
    """Extracts and stores abstract rules from episodic patterns.

    Rules are action-context-outcome generalizations learned
    through consolidation. Provides heuristic guidance for
    decision-making when episodic evidence is sparse.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def extract_rules(self, episodes: list[EpisodicEntry], current_tick: int) -> int:
        """Scan episodes for patterns meeting min_support and confidence.
        Returns number of new rules extracted."""

    def query(self, action_name: str | None = None, context: str | None = None) -> list[SemanticRule]:
        """Return matching rules, sorted by confidence descending."""

    def get_rule(self, rule_id: str) -> SemanticRule | None: ...

    def all_rules(self) -> list[SemanticRule]:
        """All stored rules sorted by confidence descending."""

    def prune_weak(self, min_confidence: float | None = None) -> int:
        """Remove rules below confidence threshold. Returns count pruned."""
```

### Extraction algorithm

```python
def extract_rules(episodes, current_tick):
    # 1. Group episodes by (action_name, scenario_name)
    # 2. For each group with count >= min_support:
    #    a. Compute mean_outcome and outcome_std
    #    b. confidence = count / total_episodes * (1 - outcome_std)
    #    c. If confidence >= threshold, create SemanticRule
    # 3. Deduplicate against existing rules (update support_count)
    # 4. Prune if exceeding max_rules (keep highest confidence)
```

**Acceptance criteria:**
- [ ] Rules extracted only when `support_count >= min_support`
- [ ] Rules with low confidence rejected
- [ ] `query()` filters by action and/or context
- [ ] Duplicate detection (same action+context updates existing rule)
- [ ] Test: 10 episodes of same action+scenario → 1 rule
- [ ] Test: mixed actions, only high-support ones extracted
- [ ] Test: prune_weak removes low-confidence rules

---

## Task 6: Prospective Memory

**File:** `src/memory/prospective.py` (~120 LOC)
**Tests:** `tests/test_memory_prospective.py` (~150 LOC)

**What:** Track active goals with priority, deadline, and progress. Provides the "what to do next" layer — feeds into working memory and future EFE pragmatic value.

### Data model

```python
class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EXPIRED = "expired"

class Goal(BaseModel):
    """An active goal with priority, deadline, and progress tracking."""

    goal_id: str
    description: str
    priority: float = 0.5           # [0, 1]
    progress: float = 0.0           # [0, 1] — 1.0 = done
    created_tick: int = 0
    deadline_tick: int | None = None
    status: GoalStatus = GoalStatus.ACTIVE
    parent_goal_id: str | None = None  # hierarchical goals
```

### Core class

```python
class ProspectiveMemory:
    """Tracks active goals with priority, deadline, and progress.

    Models the agent's intentions and commitments. Goals can be
    hierarchical (sub-goals) and expire at deadlines.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def add_goal(self, goal: Goal) -> bool:
        """Add goal if capacity allows. Returns False if at max_active."""

    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update goal progress. Auto-completes at 1.0."""

    def complete(self, goal_id: str) -> bool: ...
    def abandon(self, goal_id: str) -> bool: ...

    def active_goals(self) -> list[Goal]:
        """Active goals sorted by priority descending."""

    def check_deadlines(self, current_tick: int) -> list[Goal]:
        """Expire goals past deadline. Returns newly expired goals."""

    def top_priority(self) -> Goal | None:
        """Highest-priority active goal."""

    def sub_goals(self, parent_id: str) -> list[Goal]:
        """Child goals of a parent."""
```

**Acceptance criteria:**
- [ ] Cannot exceed `goal_max_active` active goals
- [ ] `update_progress(id, 1.0)` auto-sets status to COMPLETED
- [ ] `check_deadlines()` expires overdue goals
- [ ] `active_goals()` excludes completed/abandoned/expired
- [ ] Test: add → progress → complete lifecycle
- [ ] Test: deadline expiration
- [ ] Test: hierarchical sub-goals

---

## Task 7: Autobiographical Memory

**File:** `src/memory/autobiographical.py` (~140 LOC)
**Tests:** `tests/test_memory_autobiographical.py` (~150 LOC)

**What:** Store identity-defining events — turning points where the self-model shifted significantly, high-emotional-significance episodes, or milestones. Links to self-model for narrative identity.

### Data model

```python
class AutobiographicalEvent(BaseModel):
    """A significant life event in the agent's narrative identity."""

    event_id: str
    tick: int
    description: str
    event_type: str              # "turning_point", "milestone", "crisis", "revelation"
    emotional_valence: float     # [-1, 1]
    identity_dimensions_affected: list[str]  # e.g., ["O", "N"]
    self_model_delta: float      # magnitude of psi_hat change at this event
    source_episode_tick: int | None = None  # link to originating episodic entry
```

### Core class

```python
class AutobiographicalMemory:
    """Stores identity-defining events for narrative self-continuity.

    Links significant episodes to self-model changes, providing
    the agent with a coherent personal history that grounds
    identity and explains personality drift.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def record_event(self, event: AutobiographicalEvent) -> None:
        """Store autobiographical event. Prunes oldest if exceeding max."""

    def detect_turning_point(
        self,
        self_model_delta: float,
        episode: EpisodicEntry,
        dimensions_affected: list[str],
    ) -> AutobiographicalEvent | None:
        """Create a turning-point event if delta exceeds threshold."""

    def by_type(self, event_type: str) -> list[AutobiographicalEvent]:
        """Events of a specific type, chronological order."""

    def by_dimension(self, dimension_key: str) -> list[AutobiographicalEvent]:
        """Events affecting a specific personality dimension."""

    def narrative_summary(self, n: int = 5) -> list[AutobiographicalEvent]:
        """Top-n most significant events by |self_model_delta|."""

    def all_events(self) -> list[AutobiographicalEvent]: ...
```

### Turning point detection

Called from `SelfModel.update()` when `update_magnitude > threshold`:
```python
if update_magnitude > params.autobio_significance_threshold:
    event = autobio.detect_turning_point(update_magnitude, last_episode, dims_changed)
```

**Acceptance criteria:**
- [ ] Events stored chronologically
- [ ] `detect_turning_point()` returns None below threshold
- [ ] `narrative_summary()` returns top-n by |delta|
- [ ] `by_dimension()` filters correctly
- [ ] Max size enforced (oldest pruned)
- [ ] Test: 3 turning points recorded, 2 milestones → narrative_summary returns all 5
- [ ] Test: dimension filtering

---

## Task 8: Consolidation Engine

**File:** `src/memory/consolidation.py` (~180 LOC)
**Tests:** `tests/test_memory_consolidation.py` (~200 LOC)

**What:** Offline consolidation cycle that runs every N ticks. Performs four operations:
1. **Emotional tagging** — mark high-|valence| episodic entries for long-term retention
2. **Semantic extraction** — extract patterns from episodic clusters
3. **Procedural update** — update action success rates from recent episodes
4. **Forgetting** — decay retention and prune forgotten entries

### Core class

```python
class ConsolidationEngine:
    """Runs offline memory consolidation after configurable intervals.

    Coordinates cross-system memory operations: episodic -> semantic
    rule extraction, episodic -> procedural stat updates, and
    Ebbinghaus forgetting with emotional-significance protection.
    """

    def __init__(self, params: MemoryParams) -> None: ...

    def should_consolidate(self, current_tick: int) -> bool:
        """True when current_tick is a multiple of consolidation_interval."""

    def consolidate(
        self,
        current_tick: int,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ) -> ConsolidationReport:
        """Run the full consolidation cycle."""

    def _tag_significant(self, episodic: EpisodicMemory) -> int:
        """Mark high-valence entries with boosted retention. Returns count tagged."""

    def _extract_semantic(
        self, episodic: EpisodicMemory, semantic: SemanticMemory, tick: int,
    ) -> int:
        """Extract rules from episodic clusters. Returns count extracted."""

    def _update_procedural(
        self, episodic: EpisodicMemory, procedural: ProceduralMemory,
    ) -> int:
        """Update procedural stats from recent episodes. Returns count updated."""

    def _apply_forgetting(self, episodic: EpisodicMemory, tick: int) -> int:
        """Decay retention and prune. Returns count pruned."""
```

### ConsolidationReport

```python
class ConsolidationReport(BaseModel):
    """Summary of a consolidation cycle."""

    tick: int
    entries_tagged: int
    rules_extracted: int
    procedural_updated: int
    entries_pruned: int
    duration_ms: float
```

### Consolidation flow

```
1. _tag_significant: entries with |valence| > threshold get retention += 0.3 (capped at 1.0)
2. _extract_semantic: group retained episodes, call semantic.extract_rules()
3. _update_procedural: for each recent episode, call procedural.record()
4. _apply_forgetting: episodic.decay_all(tick), episodic.prune_forgotten(floor)
```

**Acceptance criteria:**
- [ ] `should_consolidate()` triggers at correct intervals
- [ ] Significant episodes get retention boost before forgetting
- [ ] Semantic rules extracted from episodic clusters
- [ ] Procedural stats updated from recent outcomes
- [ ] Forgetting curve applied and low-retention entries pruned
- [ ] Test: run 50 ticks, verify consolidation triggers at tick 25 and 50
- [ ] Test: significant memories survive pruning
- [ ] Test: ConsolidationReport has correct counts

---

## Task 9: MemorySystem Facade

**File:** `src/memory/system.py` (~200 LOC)
**Tests:** `tests/test_memory_system.py` (~250 LOC)

**What:** Unified facade composing all six memory subsystems. Single entry point for the simulator to store, query, and consolidate memory.

### Core class

```python
class MemorySystem:
    """Unified multi-system memory facade.

    Composes episodic, semantic, procedural, working, prospective,
    and autobiographical memory behind a single interface. The
    simulator interacts with this facade; individual subsystems
    are accessible for specialized queries.
    """

    def __init__(self, params: MemoryParams | None = None) -> None:
        self.params = params or MemoryParams()
        self.episodic = EpisodicMemory(self.params)
        self.semantic = SemanticMemory(self.params)
        self.procedural = ProceduralMemory(self.params)
        self.working = WorkingMemory(self.params)
        self.prospective = ProspectiveMemory(self.params)
        self.autobiographical = AutobiographicalMemory(self.params)
        self._consolidation = ConsolidationEngine(self.params)

    # --- Primary interface (called by simulator) ---

    def store_tick(
        self,
        tick: int,
        scenario_name: str,
        action_name: str,
        outcome: float,
        counterfactual: float,
        state_snapshot: AgentState,
        valence: float,
    ) -> EpisodicEntry:
        """Store a tick's memory across relevant subsystems.

        1. Create and store EpisodicEntry (with causal tagging)
        2. Update procedural stats
        3. Update working memory with scenario context
        """

    def maybe_consolidate(self, current_tick: int) -> ConsolidationReport | None:
        """Run consolidation if interval elapsed. Returns report or None."""

    # --- Backward-compatible MemoryBank interface ---
    # (delegates to self.episodic)

    def store(self, entry: EpisodicEntry) -> None: ...
    def recent(self, n: int) -> list[EpisodicEntry]: ...
    def mean_outcome(self, window: int = 10) -> float: ...
    def mean_valence(self, window: int = 10) -> float: ...
    def mean_arousal(self, window: int = 10) -> float: ...
    def peak_valence(self, window: int = 50) -> float: ...
    def total_regret(self, window: int = 10) -> float: ...
    def consecutive_failures(self) -> int: ...
    def outcome_variance(self, window: int = 10) -> float: ...
    def action_outcome_variance(self, action_name: str, window: int = 50) -> float | None: ...
    def __len__(self) -> int: ...

    @property
    def entries(self) -> list[EpisodicEntry]: ...
```

### Key behavior

- `store_tick()` is the new primary entry point — stores episodic + updates procedural + updates working memory in one call
- All existing `MemoryBank` query methods delegate to `self.episodic`
- `maybe_consolidate()` called at end of each tick by the simulator
- Subsystems accessible directly via `memory_system.semantic.query(...)` etc.

**Acceptance criteria:**
- [ ] `store_tick()` creates entry in episodic + updates procedural
- [ ] All backward-compat methods delegate correctly
- [ ] `maybe_consolidate()` triggers at correct intervals
- [ ] `__len__` matches episodic entry count
- [ ] Test: full lifecycle — 50 ticks of store_tick + consolidation
- [ ] Test: backward-compat — all existing MemoryBank test assertions pass

---

## Task 10: Backward-Compatible Migration

**File:** `src/temporal/memory.py` (~120 LOC, modified)
**Tests:** Existing `tests/test_temporal.py` memory tests must pass unchanged

**What:** Transform the existing `MemoryBank` into a thin shim that delegates to `MemorySystem`. This ensures zero breaking changes for all existing callers (simulator, EFE engine, emotion detectors, System 2).

### Strategy

```python
# src/temporal/memory.py — after migration

"""Episodic memory: backward-compatible interface delegating to MemorySystem."""

from src.memory.episodic import EpisodicEntry
from src.memory.system import MemorySystem
from src.temporal.state import AgentState


# Preserve the old name as an alias for backward compatibility
MemoryEntry = EpisodicEntry


class MemoryBank:
    """Backward-compatible memory interface.

    Delegates all operations to the underlying MemorySystem.
    Existing callers (simulator, EFE, detectors, System 2)
    continue working without modification.
    """

    def __init__(self, max_size: int = 500) -> None:
        from src.memory.params import MemoryParams
        params = MemoryParams(episodic_max_size=max_size)
        self._system = MemorySystem(params=params)

    @property
    def system(self) -> MemorySystem:
        """Access the underlying MemorySystem for advanced queries."""
        return self._system

    # All existing methods delegate to self._system.*
    def store(self, entry: MemoryEntry) -> None: ...
    def recent(self, n: int) -> list[MemoryEntry]: ...
    # ... etc (all 8 query methods)
```

### Migration rules

1. `MemoryEntry` becomes an alias for `EpisodicEntry`
2. `MemoryBank.__init__` constructs a `MemorySystem` internally
3. All query methods delegate to `self._system.episodic.*`
4. New `.system` property exposes the full `MemorySystem` for callers that want advanced features
5. **No changes** to any caller signatures — all imports and types preserved

**Acceptance criteria:**
- [ ] `from src.temporal.memory import MemoryBank, MemoryEntry` still works
- [ ] All existing `MemoryBank` tests pass without modification
- [ ] `MemoryEntry` fields are a superset of the old fields (new fields have defaults)
- [ ] `MemoryBank.system` exposes `MemorySystem` for opt-in advanced usage
- [ ] mypy passes with no new errors

---

## Task 11: Integration Wiring

**Files modified:** `simulator.py`, `epistemic.py`, `system2.py`, `self_model/model.py`
**Tests:** `tests/test_memory_system.py` integration section (~50 LOC addition)

### 11a: Simulator consolidation trigger

**File:** `src/temporal/simulator.py` (~10 LOC delta)

After `self._store_memory(...)` in `tick()`, add:

```python
# After line 274 (self._store_memory call)
if hasattr(self.memory, 'system'):
    report = self.memory.system.maybe_consolidate(self._tick_counter)
    if report:
        _log.info("consolidation_complete", **report.model_dump())
```

This is a zero-risk addition — the `hasattr` guard means existing `MemoryBank` instances without `MemorySystem` still work.

### 11b: EFE epistemic value enhancement

**File:** `src/efe/epistemic.py` (~15 LOC delta)

Add an optional procedural memory path:

```python
def compute_epistemic_value(
    action_name: str,
    memory: MemoryBank,
    window: int = 50,
    default: float = 1.0,
) -> float:
    """Outcome variance for a specific action from recent memory.

    When procedural memory is available (via MemorySystem), uses
    the inverse of attempt count as an exploration bonus, combined
    with outcome variance.
    """
    # New: check for procedural memory
    if hasattr(memory, 'system'):
        procedural = memory.system.procedural
        stats = procedural.action_stats(action_name)
        if stats:
            # More attempts -> lower epistemic value (less to learn)
            total_attempts = sum(r.attempts for r in stats)
            if total_attempts >= 3:
                # Blend: 70% variance + 30% exploration bonus
                variance = memory.action_outcome_variance(action_name, window)
                exploration = 1.0 / (1.0 + total_attempts * 0.1)
                return 0.7 * (variance or default) + 0.3 * exploration

    # Fallback: existing behavior
    recent = memory.recent(window)
    outcomes = [m.outcome for m in recent if m.action_name == action_name]
    if len(outcomes) < 2:
        return default
    return float(np.var(outcomes))
```

### 11c: System 2 trajectory enrichment

**File:** `src/temporal/system2.py` (~15 LOC delta)

Enrich `build_trajectory_window()` with semantic context when available:

```python
@staticmethod
def build_trajectory_window(
    memory_entries: list[Any],
    state: Any,
    memory_system: Any | None = None,  # NEW: optional MemorySystem
) -> list[dict[str, Any]]:
    """Extract trajectory window with optional semantic enrichment."""
    window = [
        {
            "state": list(state.to_array()) if hasattr(state, "to_array") else [],
            "outcome": e.outcome,
            "action": e.action_name,
        }
        for e in memory_entries
    ]

    # Enrich with semantic rules if available
    if memory_system is not None:
        rules = memory_system.semantic.all_rules()
        if rules:
            window.append({
                "semantic_context": [
                    {"pattern": r.pattern, "confidence": r.confidence}
                    for r in rules[:5]  # top 5 rules
                ],
            })

    return window
```

### 11d: Self-model autobiographical hook

**File:** `src/self_model/model.py` (~10 LOC delta)

Add optional autobiographical recording in `update()`:

```python
# After line 137 (sustained_coherence_threat check)
if (
    self._autobiographical is not None
    and update_mag > self._autobio_threshold
):
    self._autobiographical.detect_turning_point(
        self_model_delta=update_mag,
        episode=self._last_episode,
        dimensions_affected=self._changed_dimensions(delta),
    )
```

The `_autobiographical` and `_last_episode` are set via new optional params on `__init__` with `None` defaults — zero breaking change.

**Acceptance criteria:**
- [ ] Simulator triggers consolidation at correct intervals
- [ ] EFE uses procedural memory when available, falls back to variance
- [ ] System 2 includes semantic context in trajectory window
- [ ] Self-model records turning points in autobiographical memory
- [ ] All existing tests pass without modification
- [ ] No import cycles introduced
- [ ] mypy strict passes across all modified files

---

## Verification Checklist

After all 11 tasks, run:

```bash
# Full test suite
cd backend && uv run pytest tests/ -v --tb=short

# Coverage check
uv run pytest tests/ --cov=src --cov-fail-under=80

# Type checking
uv run mypy src/ --strict

# Linting
uv run ruff check src/ tests/

# Specific memory tests
uv run pytest tests/test_memory_*.py -v

# Backward compatibility regression
uv run pytest tests/test_temporal.py -k "memory" -v
uv run pytest tests/test_efe.py -v
uv run pytest tests/test_constructed_emotion.py -v
```

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| Breaking existing MemoryBank callers | Task 10: `MemoryEntry` alias + `MemoryBank` shim with identical interface |
| Import cycles (`memory` ↔ `temporal`) | New `src/memory/` package has no imports from `src/temporal/` (only `AgentState` via TYPE_CHECKING) |
| LOC limit violations | Each file budgeted under 200 LOC; facade at 200 is highest |
| Consolidation performance | Runs every 25 ticks (configurable), not every tick; negligible overhead |
| Semantic extraction false patterns | `min_support=5` and `confidence_threshold=0.7` prevent spurious rules |
| Memory bloat | Forgetting curve + max sizes enforce bounded memory usage |

---

## Non-Goals (Explicitly Deferred)

- **Emotion regulation (L3)** — uses memory but is a separate layer
- **Prospective cognition / forward simulation (L4)** — depends on this work but is a separate feature
- **Metacognitive Note-Assess-Guide loop (L2)** — will consume working memory but is separate
- **LLM-backed semantic extraction** — start with heuristic rules; LLM upgrade is a follow-up
- **Distributed memory / persistence** — in-memory only for now; SQLite persistence is a separate concern
