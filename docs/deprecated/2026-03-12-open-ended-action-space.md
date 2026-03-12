# Open-Ended Action Space Implementation Plan

> **Execution note:** Use the local `executing-plans` skill to implement this plan task-by-task.

**Goal:** Replace the finite, predefined action space (Action objects with explicit modifier vectors) with an open-ended action space supporting tool calls, API requests, and free-text generation — while preserving the full precision hierarchy, EFE, self-evidencing, and Boltzmann decision machinery.

**Architecture:** Introduce a two-phase action pipeline: (1) **propose** — an LLM-backed ActionProposer generates candidate actions from context, tools, and state; (2) **encode** — an ActionEncoder maps each candidate to a personality-dimension modifier vector so existing decision/EFE/self-evidencing math works unchanged. After selection, an ActionExecutor dispatches the chosen action (tool call, HTTP request, or text generation) and returns a typed result. Classic predefined actions remain a first-class path (zero-encoding cost, full backward compatibility).

**Tech Stack:** Python 3.11+, Pydantic, FastAPI, structlog, numpy, pytest, ruff, mypy

**LOC Budget:**

| New/Modified File | Estimated LOC | Notes |
|---|---|---|
| `src/action_space/proposal.py` | ~120 | ActionProposal + subtypes |
| `src/action_space/encoder.py` | ~150 | LLM + heuristic modifier encoding |
| `src/action_space/proposer.py` | ~180 | Context-aware action generation |
| `src/action_space/executor.py` | ~160 | Dispatch + result model |
| `src/action_space/registry.py` | ~100 | Tool/capability registry |
| `src/action_space/caching_encoder.py` | ~80 | LRU cache decorator over any encoder backend |
| `src/action_space/params.py` | ~40 | Hyperparameters |
| `src/personality/vectors.py` | ~90 (from 77) | Add `Action.from_proposal()` class method |
| `src/personality/decision.py` | ~175 (from 168) | Accept ActionProposal alongside Action |
| `src/shared/protocols.py` | ~100 (from 78) | Add ActionEncoderProtocol, ActionExecutorProtocol |
| `src/sdk/__init__.py` | ~295 (from 283) | Add `propose_actions()`, `execute_action()` |
| `src/llm/agent_runtime.py` | ~250 (from 229) | Add `encode_action()`, `propose_actions()` |
| `src/llm/schemas.py` | ~115 (from 96) | Add ActionProposalLLM, ActionEncoding schemas |
| `src/temporal/simulator.py` | ~330 (from 314) | Wire dynamic action proposal into tick |
| Tests (7 new files) | ~200 each | Full TDD coverage |

---

## Design Principles

### 1. The Modifier Vector is the Universal Currency

The existing decision machinery — `U(ψ, s, a) = activations · modifiers`, Boltzmann softmax, EFE pragmatic/epistemic decomposition, self-evidencing precision weights — all operate on **modifier vectors**. Rather than replacing this, we preserve it as the internal representation and add an **encoding layer** that projects any open-ended action into the modifier space.

This means:
- `DecisionEngine.decide()` signature stays the same
- `EFEEngine` keeps working
- `SelfEvidencingModulator` keeps working
- `SelfModel.update()` keeps working
- All 524 existing tests keep passing

### 2. Separation: Propose → Encode → Decide → Execute

```
Context + State + Tools
        ↓
   ActionProposer          ← LLM generates candidates
        ↓
   [ActionProposal, ...]   ← typed proposals (tool/api/text/classic)
        ↓
   ActionEncoder           ← maps each proposal to modifier vector
        ↓
   [Action, ...]           ← standard Action objects with modifiers
        ↓
   DecisionEngine.decide() ← unchanged Boltzmann / EFE selection
        ↓
   chosen Action + linked ActionProposal
        ↓
   ActionExecutor          ← dispatches tool call / API / text
        ↓
   ActionResult            ← typed result with outcome signal
```

### 3. Classic Actions = Zero-Cost Path

Predefined `Action` objects with explicit modifiers bypass proposer and encoder entirely. The `from_proposal()` factory on `Action` creates a standard `Action` from an encoded proposal, unifying the two paths at the decision engine boundary.

### 4. Graceful Degradation

When no LLM is configured:
- Proposer falls back to classic predefined actions
- Encoder uses heuristic keyword-based modifier estimation
- Executor for text actions returns a template response

---

## Phase D1: Action Proposal Types and Registry

### Task 1: ActionProposal Pydantic Models

**Files:**
- Create: `backend/src/action_space/__init__.py`
- Create: `backend/src/action_space/proposal.py`
- Test: `backend/tests/test_action_proposal.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_proposal.py
"""Tests for action proposal types."""

import pytest

from src.action_space.proposal import (
    ActionProposal,
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)


class TestToolActionProposal:
    """ToolActionProposal validation."""

    def test_valid_tool_call(self) -> None:
        proposal = ToolActionProposal(
            name="search_web",
            description="Search for recent papers on active inference",
            tool_name="web_search",
            tool_args={"query": "active inference precision 2025"},
        )
        assert proposal.kind == "tool"
        assert proposal.tool_name == "web_search"

    def test_tool_name_required(self) -> None:
        with pytest.raises(ValueError):
            ToolActionProposal(
                name="search",
                description="search",
                tool_name="",
                tool_args={},
            )


class TestApiActionProposal:
    """ApiActionProposal validation."""

    def test_valid_api_request(self) -> None:
        proposal = ApiActionProposal(
            name="fetch_weather",
            description="Get current weather data",
            method="GET",
            url="https://api.weather.example/current",
            headers={"Authorization": "Bearer tok"},
        )
        assert proposal.kind == "api"
        assert proposal.method == "GET"

    def test_method_constrained(self) -> None:
        with pytest.raises(ValueError):
            ApiActionProposal(
                name="bad",
                description="bad",
                method="PATCH",
                url="https://example.com",
            )


class TestTextActionProposal:
    """TextActionProposal validation."""

    def test_valid_text_generation(self) -> None:
        proposal = TextActionProposal(
            name="explain_concept",
            description="Explain active inference to the user",
            intent="explain",
            prompt_hint="Describe precision weighting in plain language",
        )
        assert proposal.kind == "text"

    def test_intent_required(self) -> None:
        with pytest.raises(ValueError):
            TextActionProposal(
                name="bad",
                description="bad",
                intent="",
            )


class TestClassicActionProposal:
    """ClassicActionProposal wraps predefined modifiers."""

    def test_wraps_existing_modifiers(self) -> None:
        proposal = ClassicActionProposal(
            name="bold",
            description="Take a bold approach",
            modifiers={"O": 1.0, "R": 0.8, "N": -0.3},
        )
        assert proposal.kind == "classic"
        assert proposal.modifiers["O"] == 1.0

    def test_modifier_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            ClassicActionProposal(
                name="bad",
                description="bad",
                modifiers={"O": 1.5},
            )


class TestActionProposalDiscriminator:
    """Union type resolves correctly."""

    def test_discriminator_routing(self) -> None:
        data = {
            "kind": "tool",
            "name": "search",
            "description": "search",
            "tool_name": "web_search",
            "tool_args": {"query": "test"},
        }
        proposal = ActionProposal.model_validate(data)
        assert isinstance(proposal.root, ToolActionProposal)
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_proposal.py -v`
Expected: FAIL (module not found)

**Step 3: Implement the models**

```python
# backend/src/action_space/__init__.py
"""Open-ended action space: propose, encode, decide, execute."""

# backend/src/action_space/proposal.py
"""Action proposal types for open-ended action spaces.

Each proposal kind represents a different action modality:
- tool: invoke a registered tool/capability
- api: make an HTTP request to an external service
- text: generate free-text output
- classic: wrap predefined modifier vectors (backward compat)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel, field_validator


class _ProposalBase(BaseModel):
    """Shared fields for all action proposals."""

    name: str = Field(min_length=1)
    description: str = ""
    rationale: str = ""


class ToolActionProposal(_ProposalBase):
    """Invoke a registered tool or capability."""

    kind: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, object] = Field(default_factory=dict)
    timeout_ms: int = Field(default=30_000, gt=0)


class ApiActionProposal(_ProposalBase):
    """Make an HTTP request to an external service."""

    kind: Literal["api"] = "api"
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, object] | None = None


class TextActionProposal(_ProposalBase):
    """Generate free-text output."""

    kind: Literal["text"] = "text"
    intent: str = Field(min_length=1)
    prompt_hint: str = ""
    max_tokens: int = Field(default=1024, gt=0)


class ClassicActionProposal(_ProposalBase):
    """Wrap predefined personality-dimension modifiers (backward compat)."""

    kind: Literal["classic"] = "classic"
    modifiers: dict[str, float]

    @field_validator("modifiers")
    @classmethod
    def _check_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not -1.0 <= val <= 1.0:
                msg = f"{key}={val} outside [-1, 1]"
                raise ValueError(msg)
        return v


_ProposalUnion = Annotated[
    ToolActionProposal | ApiActionProposal | TextActionProposal | ClassicActionProposal,
    Field(discriminator="kind"),
]


class ActionProposal(RootModel[_ProposalUnion]):
    """Discriminated union over action proposal types."""

    @property
    def kind(self) -> str:
        """Delegate to inner model."""
        return self.root.kind

    @property
    def name(self) -> str:
        """Delegate to inner model."""
        return self.root.name

    @property
    def description(self) -> str:
        """Delegate to inner model."""
        return self.root.description
```

**Step 4: Run tests to verify pass**

Run: `cd backend && uv run pytest tests/test_action_proposal.py -v`
Expected: PASS

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src/action_space tests/test_action_proposal.py && uv run mypy src/action_space`
Expected: No errors

**Step 6: Commit**

```bash
git add src/action_space/ tests/test_action_proposal.py
git commit -m "feat(action-space): add ActionProposal types for open-ended actions"
```

---

### Task 2: Tool Registry

**Files:**
- Create: `backend/src/action_space/registry.py`
- Test: `backend/tests/test_action_registry.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_registry.py
"""Tests for the tool/capability registry."""

import pytest

from src.action_space.registry import ToolCapability, ToolRegistry


class TestToolRegistry:
    """ToolRegistry manages available tools."""

    def test_register_and_lookup(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(
            name="web_search",
            description="Search the web for information",
            parameter_schema={"query": {"type": "string"}},
            personality_hint={"O": 0.7, "E": 0.3},
        )
        registry.register(cap)
        assert registry.get("web_search") is cap

    def test_lookup_missing_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        cap1 = ToolCapability(
            name="search",
            description="search",
            parameter_schema={},
        )
        cap2 = ToolCapability(
            name="calculate",
            description="calculate",
            parameter_schema={},
        )
        registry.register(cap1)
        registry.register(cap2)
        names = [t.name for t in registry.list_tools()]
        assert "search" in names
        assert "calculate" in names

    def test_duplicate_name_raises(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(name="a", description="a", parameter_schema={})
        registry.register(cap)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(cap)

    def test_personality_hint_default_empty(self) -> None:
        cap = ToolCapability(name="t", description="t", parameter_schema={})
        assert cap.personality_hint == {}

    def test_to_prompt_context(self) -> None:
        registry = ToolRegistry()
        cap = ToolCapability(
            name="search",
            description="Search for things",
            parameter_schema={"query": {"type": "string"}},
        )
        registry.register(cap)
        ctx = registry.to_prompt_context()
        assert "search" in ctx
        assert "query" in ctx
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_registry.py -v`
Expected: FAIL

**Step 3: Implement**

```python
# backend/src/action_space/registry.py
"""Tool and capability registry for open-ended action spaces.

Each tool declares its parameter schema and an optional personality hint
mapping personality dimensions to affinity scores. The hint guides the
ActionEncoder when estimating modifier vectors for tool-based actions.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.shared.logging import get_logger

_log = get_logger(module="action_space.registry")


class ToolCapability(BaseModel):
    """A registered tool or capability the agent can invoke."""

    name: str = Field(min_length=1)
    description: str
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    personality_hint: dict[str, float] = Field(default_factory=dict)


class ToolRegistry:
    """Manages the set of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolCapability] = {}

    def register(self, capability: ToolCapability) -> None:
        """Register a tool. Raises ValueError on duplicate names."""
        if capability.name in self._tools:
            msg = f"Tool '{capability.name}' already registered"
            raise ValueError(msg)
        self._tools[capability.name] = capability
        _log.debug("tool_registered", tool=capability.name)

    def get(self, name: str) -> ToolCapability:
        """Look up a tool by name. Raises KeyError if missing."""
        return self._tools[name]

    def list_tools(self) -> list[ToolCapability]:
        """Return all registered tools."""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def to_prompt_context(self) -> str:
        """Serialize registry as a prompt-friendly string for LLM proposers."""
        entries = []
        for tool in self._tools.values():
            entries.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameter_schema,
                }
            )
        return json.dumps(entries, indent=2)
```

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_action_registry.py -v`
Expected: PASS

**Step 5: Lint**

Run: `cd backend && uv run ruff check src/action_space/registry.py tests/test_action_registry.py && uv run mypy src/action_space`
Expected: No errors

**Step 6: Commit**

```bash
git add src/action_space/registry.py tests/test_action_registry.py
git commit -m "feat(action-space): add ToolRegistry for capability management"
```

---

## Phase D2: Action Encoder

### Task 3: ActionEncoder — Map Proposals to Modifier Vectors

**Files:**
- Create: `backend/src/action_space/encoder.py`
- Create: `backend/src/action_space/params.py`
- Modify: `backend/src/shared/protocols.py:60-78` (add ActionEncoderProtocol)
- Modify: `backend/src/llm/schemas.py:79-96` (add ActionEncoding schema)
- Modify: `backend/src/llm/agent_runtime.py:194-228` (add encode_action method)
- Test: `backend/tests/test_action_encoder.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_encoder.py
"""Tests for action-to-modifier encoding."""

import numpy as np
import pytest

from src.action_space.encoder import ActionEncoder, HeuristicEncoderBackend
from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)
from src.action_space.registry import ToolCapability, ToolRegistry
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action


class TestClassicEncoding:
    """Classic proposals pass through without LLM call."""

    def test_classic_becomes_action_directly(self) -> None:
        registry = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=registry,
            backend=HeuristicEncoderBackend(),
        )
        proposal = ClassicActionProposal(
            name="bold",
            description="bold move",
            modifiers={"O": 1.0, "R": 0.8},
        )
        action = encoder.encode(proposal)
        assert isinstance(action, Action)
        assert action.name == "bold"
        o_idx = registry.index("O")
        assert action.modifiers[o_idx] == pytest.approx(1.0)

    def test_classic_preserves_all_modifiers(self) -> None:
        registry = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=registry,
            backend=HeuristicEncoderBackend(),
        )
        proposal = ClassicActionProposal(
            name="safe",
            description="safe choice",
            modifiers={"C": 0.9, "T": 0.8, "N": -0.2},
        )
        action = encoder.encode(proposal)
        c_idx = registry.index("C")
        t_idx = registry.index("T")
        n_idx = registry.index("N")
        assert action.modifiers[c_idx] == pytest.approx(0.9)
        assert action.modifiers[t_idx] == pytest.approx(0.8)
        assert action.modifiers[n_idx] == pytest.approx(-0.2)


class TestToolEncoding:
    """Tool proposals encode via personality hint + heuristic."""

    def test_tool_with_personality_hint(self) -> None:
        dim_reg = DimensionRegistry()
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search the web",
                parameter_schema={"query": {"type": "string"}},
                personality_hint={"O": 0.8, "E": 0.4},
            )
        )
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(tool_registry=tool_reg),
        )
        proposal = ToolActionProposal(
            name="search_papers",
            description="search for papers",
            tool_name="web_search",
            tool_args={"query": "active inference"},
        )
        action = encoder.encode(proposal)
        o_idx = dim_reg.index("O")
        assert action.modifiers[o_idx] == pytest.approx(0.8)

    def test_tool_without_hint_uses_defaults(self) -> None:
        dim_reg = DimensionRegistry()
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="noop",
                description="does nothing",
                parameter_schema={},
            )
        )
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(tool_registry=tool_reg),
        )
        proposal = ToolActionProposal(
            name="do_nothing",
            description="nothing",
            tool_name="noop",
            tool_args={},
        )
        action = encoder.encode(proposal)
        assert action.modifiers.shape == (dim_reg.size,)
        assert np.all(np.abs(action.modifiers) <= 1.0)


class TestTextEncoding:
    """Text proposals encode via heuristic intent mapping."""

    def test_explain_intent_maps_to_openness(self) -> None:
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposal = TextActionProposal(
            name="explain",
            description="explain a concept",
            intent="explain",
        )
        action = encoder.encode(proposal)
        o_idx = dim_reg.index("O")
        assert action.modifiers[o_idx] > 0.3

    def test_output_always_bounded(self) -> None:
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposal = TextActionProposal(
            name="rant",
            description="go on a rant",
            intent="express_frustration",
        )
        action = encoder.encode(proposal)
        assert np.all(action.modifiers >= -1.0)
        assert np.all(action.modifiers <= 1.0)


class TestBatchEncoding:
    """Batch encoding for multiple proposals."""

    def test_encode_batch(self) -> None:
        dim_reg = DimensionRegistry()
        encoder = ActionEncoder(
            dimension_registry=dim_reg,
            backend=HeuristicEncoderBackend(),
        )
        proposals = [
            ClassicActionProposal(name="a", description="a", modifiers={"O": 0.5}),
            ClassicActionProposal(name="b", description="b", modifiers={"C": 0.9}),
        ]
        actions = encoder.encode_batch(proposals)
        assert len(actions) == 2
        assert actions[0].name == "a"
        assert actions[1].name == "b"
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_encoder.py -v`
Expected: FAIL

**Step 3: Add ActionEncoderProtocol to protocols.py**

Add to `backend/src/shared/protocols.py` after `System2RuntimeProtocol`:

```python
class ActionEncoderProtocol(Protocol):
    """Structural interface for action-to-modifier encoding."""

    def encode_modifiers(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Return estimated personality-dimension modifiers in [-1, 1]."""
        ...
```

**Step 4: Add ActionEncoding to llm/schemas.py**

Add to `backend/src/llm/schemas.py` after `EmotionConstructor`:

```python
class ActionEncoding(BaseModel):
    """LLM-estimated personality-dimension modifiers for an open-ended action."""

    modifiers: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
```

**Step 5: Implement params.py**

```python
# backend/src/action_space/params.py
"""Hyperparameters for action space encoding and proposal."""

from pydantic import BaseModel, Field


class ActionEncoderParams(BaseModel):
    """Controls encoding behavior."""

    default_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)
    heuristic_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
```

**Step 6: Implement encoder.py**

```python
# backend/src/action_space/encoder.py
"""Action encoder: maps open-ended proposals to personality-dimension modifiers.

Classic proposals pass through directly. Tool, API, and text proposals
are encoded via a pluggable backend (heuristic or LLM-backed).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.action_space.params import ActionEncoderParams
from src.action_space.proposal import (
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action
from src.sdk.builders import build_action
from src.shared.logging import get_logger

_log = get_logger(module="action_space.encoder")

# Heuristic intent-to-modifier mappings for text actions.
_INTENT_HINTS: dict[str, dict[str, float]] = {
    "explain": {"O": 0.6, "A": 0.4, "E": 0.3},
    "persuade": {"E": 0.7, "O": 0.5, "A": -0.2},
    "comfort": {"A": 0.8, "E": 0.3, "N": -0.4},
    "challenge": {"O": 0.6, "E": 0.5, "A": -0.4, "N": 0.3},
    "withdraw": {"E": -0.6, "N": 0.5, "R": -0.3},
    "organize": {"C": 0.8, "T": 0.4},
    "create": {"O": 0.9, "I": 0.5, "C": -0.2},
    "analyze": {"O": 0.5, "C": 0.7, "I": 0.4},
}


class EncoderBackend(Protocol):
    """Pluggable backend for modifier estimation."""

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Return estimated modifiers for a non-classic proposal."""
        ...


class HeuristicEncoderBackend:
    """Deterministic heuristic encoding (no LLM calls).

    Uses tool personality hints and intent-to-modifier lookup tables.
    Falls back to zero modifiers for unknown actions.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self._tools = tool_registry

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Estimate modifiers from tool hints or intent keywords."""
        if isinstance(proposal, ToolActionProposal):
            return self._encode_tool(proposal)
        if isinstance(proposal, ApiActionProposal):
            return self._encode_api(proposal)
        if isinstance(proposal, TextActionProposal):
            return self._encode_text(proposal)
        return {}

    def _encode_tool(self, proposal: ToolActionProposal) -> dict[str, float]:
        if self._tools and self._tools.has(proposal.tool_name):
            return dict(self._tools.get(proposal.tool_name).personality_hint)
        return {"O": 0.3}

    def _encode_api(self, proposal: ApiActionProposal) -> dict[str, float]:
        base: dict[str, float] = {"O": 0.4, "C": 0.3}
        if proposal.method in ("POST", "PUT", "DELETE"):
            base["E"] = 0.3
        return base

    def _encode_text(self, proposal: TextActionProposal) -> dict[str, float]:
        intent = proposal.intent.lower().strip()
        for key, hint in _INTENT_HINTS.items():
            if key in intent:
                return dict(hint)
        return {"O": 0.3, "E": 0.2}


class ActionEncoder:
    """Maps ActionProposals to Action objects with modifier vectors.

    Classic proposals pass through directly (zero encoding cost).
    Other kinds delegate to an EncoderBackend.
    """

    def __init__(
        self,
        dimension_registry: DimensionRegistry,
        backend: EncoderBackend,
        params: ActionEncoderParams = ActionEncoderParams(),
    ) -> None:
        self._dim_reg = dimension_registry
        self._backend = backend
        self._params = params

    def encode(self, proposal: _ProposalBase) -> Action:
        """Encode a single proposal into an Action with modifier vector."""
        if isinstance(proposal, ClassicActionProposal):
            return build_action(
                proposal.name,
                proposal.modifiers,
                self._dim_reg,
                description=proposal.description,
            )

        modifiers = self._backend.estimate(proposal)
        _log.debug(
            "action_encoded",
            name=proposal.name,
            kind=getattr(proposal, "kind", "unknown"),
            modifiers=modifiers,
        )
        return build_action(
            proposal.name,
            modifiers,
            self._dim_reg,
            description=proposal.description,
        )

    def encode_batch(self, proposals: list[_ProposalBase]) -> list[Action]:
        """Encode multiple proposals into Action objects."""
        return [self.encode(p) for p in proposals]
```

**Step 7: Run tests**

Run: `cd backend && uv run pytest tests/test_action_encoder.py -v`
Expected: PASS

**Step 8: Run full suite**

Run: `cd backend && uv run pytest -x -q`
Expected: 524+ passed

**Step 9: Lint**

Run: `cd backend && uv run ruff check src/action_space tests/test_action_encoder.py && uv run mypy src/action_space`

**Step 10: Commit**

```bash
git add src/action_space/encoder.py src/action_space/params.py src/shared/protocols.py src/llm/schemas.py tests/test_action_encoder.py
git commit -m "feat(action-space): add ActionEncoder with heuristic backend"
```

---

## Phase D3: Action Proposer

### Task 4: LLM-Backed Action Proposer

**Files:**
- Create: `backend/src/action_space/proposer.py`
- Modify: `backend/src/llm/agent_runtime.py` (add `propose_actions` method)
- Modify: `backend/src/llm/schemas.py` (add `ActionSetProposal` schema)
- Test: `backend/tests/test_action_proposer.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_proposer.py
"""Tests for context-aware action proposal."""

from src.action_space.proposal import ClassicActionProposal, ToolActionProposal
from src.action_space.proposer import ActionProposer, StaticProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry


class TestStaticProposerBackend:
    """Static backend returns predefined actions (fallback path)."""

    def test_returns_classic_proposals(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
            ClassicActionProposal(
                name="safe", description="safe", modifiers={"C": 0.9}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposals = backend.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 2
        assert proposals[0].name == "bold"


class TestActionProposer:
    """ActionProposer merges tool-based and backend proposals."""

    def test_includes_tool_proposals_from_registry(self) -> None:
        tool_reg = ToolRegistry()
        tool_reg.register(
            ToolCapability(
                name="web_search",
                description="search the web",
                parameter_schema={"query": {"type": "string"}},
            )
        )
        classics = [
            ClassicActionProposal(
                name="safe", description="safe", modifiers={"C": 0.9}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(
            backend=backend,
            tool_registry=tool_reg,
        )
        proposals = proposer.propose(
            state={"energy": 0.8},
            trajectory=[],
            goals=["find information"],
        )
        kinds = [getattr(p, "kind", None) for p in proposals]
        assert "tool" in kinds
        assert "classic" in kinds

    def test_no_tools_returns_backend_only(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 1
        assert proposals[0].name == "bold"

    def test_max_proposals_limit(self) -> None:
        classics = [
            ClassicActionProposal(
                name=f"a{i}", description="", modifiers={"O": 0.1}
            )
            for i in range(20)
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend, max_proposals=5)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        assert len(proposals) == 5

    def test_always_includes_withdraw(self) -> None:
        classics = [
            ClassicActionProposal(
                name="bold", description="bold", modifiers={"O": 1.0}
            ),
        ]
        backend = StaticProposerBackend(defaults=classics)
        proposer = ActionProposer(backend=backend, include_withdraw=True)
        proposals = proposer.propose(state={}, trajectory=[], goals=[])
        names = [p.name for p in proposals]
        assert "Withdraw" in names
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_proposer.py -v`
Expected: FAIL

**Step 3: Add ActionSetProposal to llm/schemas.py**

Add after `ActionEncoding`:

```python
class ActionSetProposal(BaseModel):
    """LLM-proposed set of candidate actions given context."""

    actions: list[dict[str, object]]
    rationale: str = ""
```

**Step 4: Implement proposer.py**

```python
# backend/src/action_space/proposer.py
"""Context-aware action proposal for open-ended action spaces.

The ActionProposer generates candidate actions from:
1. A pluggable backend (static defaults or LLM-generated)
2. Available tools from the ToolRegistry
3. A mandatory Withdraw action (when enabled)
"""

from __future__ import annotations

from typing import Any, Protocol

from src.action_space.proposal import (
    ClassicActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.shared.logging import get_logger

_log = get_logger(module="action_space.proposer")

_WITHDRAW = ClassicActionProposal(
    name="Withdraw",
    description="Disengage from the current scenario",
    modifiers={"E": -0.8, "R": -0.3},
)


class ProposerBackend(Protocol):
    """Pluggable backend for generating action candidates."""

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Return candidate proposals given current context."""
        ...


class StaticProposerBackend:
    """Returns a fixed set of classic action proposals (fallback)."""

    def __init__(self, defaults: list[_ProposalBase] | None = None) -> None:
        self._defaults = list(defaults) if defaults else []

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Return the static defaults regardless of context."""
        return list(self._defaults)


class ActionProposer:
    """Merges tool-based proposals with backend-generated candidates."""

    def __init__(
        self,
        backend: ProposerBackend,
        tool_registry: ToolRegistry | None = None,
        max_proposals: int = 10,
        include_withdraw: bool = False,
    ) -> None:
        self._backend = backend
        self._tools = tool_registry
        self._max = max_proposals
        self._withdraw = include_withdraw

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Generate candidate actions from context."""
        candidates: list[_ProposalBase] = []

        if self._tools:
            candidates.extend(self._tool_proposals())

        candidates.extend(self._backend.propose(state, trajectory, goals))

        if self._withdraw:
            candidates.append(_WITHDRAW)

        result = candidates[: self._max]

        _log.debug(
            "actions_proposed",
            count=len(result),
            kinds=[getattr(p, "kind", "unknown") for p in result],
        )
        return result

    def _tool_proposals(self) -> list[_ProposalBase]:
        """Create one proposal per registered tool with default args."""
        if not self._tools:
            return []
        proposals: list[_ProposalBase] = []
        for tool in self._tools.list_tools():
            proposals.append(
                ToolActionProposal(
                    name=f"use_{tool.name}",
                    description=tool.description,
                    tool_name=tool.name,
                    tool_args={},
                )
            )
        return proposals
```

**Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_action_proposer.py -v`
Expected: PASS

**Step 6: Lint**

Run: `cd backend && uv run ruff check src/action_space/proposer.py tests/test_action_proposer.py && uv run mypy src/action_space`

**Step 7: Commit**

```bash
git add src/action_space/proposer.py src/llm/schemas.py tests/test_action_proposer.py
git commit -m "feat(action-space): add ActionProposer with static and tool-based generation"
```

---

## Phase D4: Action Executor

### Task 5: ActionExecutor — Dispatch and Execute

**Files:**
- Create: `backend/src/action_space/executor.py`
- Test: `backend/tests/test_action_executor.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_executor.py
"""Tests for action execution dispatch."""

import pytest

from src.action_space.executor import (
    ActionExecutor,
    ActionResult,
    NoopToolHandler,
)
from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
)


class TestActionResult:
    """ActionResult carries typed execution output."""

    def test_success_result(self) -> None:
        result = ActionResult(
            success=True,
            output={"data": [1, 2, 3]},
            outcome_signal=0.6,
        )
        assert result.success
        assert result.outcome_signal == pytest.approx(0.6)

    def test_failure_result(self) -> None:
        result = ActionResult(
            success=False,
            output={},
            error="timeout",
            outcome_signal=-0.3,
        )
        assert not result.success
        assert result.error == "timeout"


class TestActionExecutor:
    """ActionExecutor dispatches to the correct handler."""

    def test_classic_actions_return_noop_result(self) -> None:
        executor = ActionExecutor()
        proposal = ClassicActionProposal(
            name="bold", description="bold", modifiers={"O": 1.0}
        )
        result = executor.execute(proposal)
        assert result.success
        assert result.outcome_signal is None

    def test_tool_action_dispatches_to_handler(self) -> None:
        handler = NoopToolHandler()
        executor = ActionExecutor(tool_handlers={"web_search": handler})
        proposal = ToolActionProposal(
            name="search",
            description="search",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        result = executor.execute(proposal)
        assert result.success

    def test_missing_tool_handler_returns_failure(self) -> None:
        executor = ActionExecutor()
        proposal = ToolActionProposal(
            name="search",
            description="search",
            tool_name="missing_tool",
            tool_args={},
        )
        result = executor.execute(proposal)
        assert not result.success
        assert "no handler" in (result.error or "").lower()

    def test_text_action_returns_placeholder(self) -> None:
        executor = ActionExecutor()
        proposal = TextActionProposal(
            name="explain",
            description="explain a concept",
            intent="explain",
        )
        result = executor.execute(proposal)
        assert result.success
        assert "text" in str(result.output).lower() or result.output is not None
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_executor.py -v`
Expected: FAIL

**Step 3: Implement executor.py**

```python
# backend/src/action_space/executor.py
"""Action executor: dispatches chosen actions for real-world effects.

Classic actions require no execution (they are personality-space abstractions).
Tool, API, and text actions are dispatched to registered handlers.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.action_space.proposal import (
    ApiActionProposal,
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.shared.logging import get_logger

_log = get_logger(module="action_space.executor")


class ActionResult(BaseModel):
    """Typed result from executing an action."""

    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    outcome_signal: float | None = None


class ToolHandler(Protocol):
    """Interface for tool execution handlers."""

    def execute(self, tool_args: dict[str, Any]) -> ActionResult:
        """Execute the tool with the given arguments."""
        ...


class NoopToolHandler:
    """Placeholder handler that always succeeds with empty output."""

    def execute(self, tool_args: dict[str, Any]) -> ActionResult:
        """Return a successful no-op result."""
        return ActionResult(success=True, output={"handler": "noop", "args": tool_args})


class ActionExecutor:
    """Dispatches action proposals to appropriate handlers."""

    def __init__(
        self,
        tool_handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        self._tool_handlers = tool_handlers or {}

    def execute(self, proposal: _ProposalBase) -> ActionResult:
        """Execute the given action proposal and return a typed result."""
        if isinstance(proposal, ClassicActionProposal):
            return self._execute_classic(proposal)
        if isinstance(proposal, ToolActionProposal):
            return self._execute_tool(proposal)
        if isinstance(proposal, ApiActionProposal):
            return self._execute_api(proposal)
        if isinstance(proposal, TextActionProposal):
            return self._execute_text(proposal)
        return ActionResult(success=False, error=f"Unknown proposal type: {type(proposal)}")

    def _execute_classic(self, proposal: ClassicActionProposal) -> ActionResult:
        _log.debug("classic_action_executed", name=proposal.name)
        return ActionResult(success=True, output={"kind": "classic"})

    def _execute_tool(self, proposal: ToolActionProposal) -> ActionResult:
        handler = self._tool_handlers.get(proposal.tool_name)
        if handler is None:
            msg = f"No handler registered for tool '{proposal.tool_name}'"
            _log.warning("tool_handler_missing", tool=proposal.tool_name)
            return ActionResult(success=False, error=msg)
        return handler.execute(proposal.tool_args)

    def _execute_api(self, proposal: ApiActionProposal) -> ActionResult:
        _log.debug("api_action_placeholder", method=proposal.method, url=proposal.url)
        return ActionResult(
            success=True,
            output={"kind": "api", "method": proposal.method, "url": proposal.url},
        )

    def _execute_text(self, proposal: TextActionProposal) -> ActionResult:
        _log.debug("text_action_placeholder", intent=proposal.intent)
        return ActionResult(
            success=True,
            output={"kind": "text", "intent": proposal.intent},
        )
```

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_action_executor.py -v`
Expected: PASS

**Step 5: Lint**

Run: `cd backend && uv run ruff check src/action_space/executor.py tests/test_action_executor.py && uv run mypy src/action_space`

**Step 6: Commit**

```bash
git add src/action_space/executor.py tests/test_action_executor.py
git commit -m "feat(action-space): add ActionExecutor with tool/api/text dispatch"
```

---

## Phase D5: SDK Integration

### Task 6: Wire Action Space into AgentSDK

**Files:**
- Modify: `backend/src/sdk/__init__.py` (add `open_ended()` factory, `propose_and_decide()`)
- Create: `backend/tests/test_sdk_open_ended.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_sdk_open_ended.py
"""Tests for open-ended action space via AgentSDK."""

from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import StaticProposerBackend
from src.sdk import AgentSDK


class TestSDKOpenEnded:
    """AgentSDK supports open-ended action proposal and decision."""

    def test_with_open_actions_factory(self) -> None:
        sdk = AgentSDK.with_open_actions()
        assert sdk is not None

    def test_propose_and_decide_with_classics(self) -> None:
        classics = [
            ClassicActionProposal(name="bold", description="bold", modifiers={"O": 1.0, "R": 0.8}),
            ClassicActionProposal(name="safe", description="safe", modifiers={"C": 0.9, "T": 0.8}),
        ]
        backend = StaticProposerBackend(defaults=classics)
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="pitch")
        result = sdk.propose_and_decide(personality, scenario)
        assert result.chosen_action in ("bold", "safe")
        assert len(result.probabilities) == 2

    def test_backward_compat_decide_unchanged(self) -> None:
        sdk = AgentSDK.default()
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        scenario = sdk.scenario({"O": 0.9}, name="test")
        actions = [
            sdk.action("bold", {"O": 1.0}),
            sdk.action("safe", {"C": 0.9}),
        ]
        result = sdk.decide(personality, scenario, actions)
        assert result.chosen_action in ("bold", "safe")

    def test_propose_and_decide_returns_proposal_metadata(self) -> None:
        classics = [
            ClassicActionProposal(name="bold", description="bold move", modifiers={"O": 1.0}),
        ]
        backend = StaticProposerBackend(defaults=classics)
        sdk = AgentSDK.with_open_actions(proposer_backend=backend)
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        scenario = sdk.scenario({"O": 0.9}, name="test")
        result = sdk.propose_and_decide(personality, scenario)
        assert result.proposals is not None
        assert len(result.proposals) >= 1
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_sdk_open_ended.py -v`
Expected: FAIL

**Step 3: Add OpenEndedDecisionResult to sdk/types.py**

Add to `backend/src/sdk/types.py`:

```python
class OpenEndedDecisionResult(BaseModel):
    """Decision result with action proposal metadata."""

    chosen_action: str
    probabilities: dict[str, float]
    utilities: dict[str, float]
    activations: dict[str, float]
    action_order: list[str]
    proposals: list[dict[str, object]] | None = None
```

**Step 4: Wire into AgentSDK.__init__.py**

Add the `with_open_actions()` class method and `propose_and_decide()` instance method. The existing `decide()` stays unchanged.

Key additions to `AgentSDK`:

```python
@classmethod
def with_open_actions(
    cls,
    proposer_backend: ProposerBackend | None = None,
    tool_registry: ToolRegistry | None = None,
    **kwargs: Any,
) -> AgentSDK:
    """Factory with open-ended action space support."""
    ...

def propose_and_decide(
    self,
    personality: PersonalityVector,
    scenario: Scenario,
    *,
    state: dict[str, Any] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    goals: list[str] | None = None,
    temperature: float = 1.0,
) -> OpenEndedDecisionResult:
    """Propose actions from context, encode, decide via Boltzmann."""
    ...
```

**Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_sdk_open_ended.py -v`
Expected: PASS

**Step 6: Run full suite**

Run: `cd backend && uv run pytest -x -q`
Expected: 524+ passed (all existing tests still pass)

**Step 7: Lint**

Run: `cd backend && uv run ruff check src/sdk tests/test_sdk_open_ended.py && uv run mypy src/sdk`

**Step 8: Commit**

```bash
git add src/sdk/__init__.py src/sdk/types.py tests/test_sdk_open_ended.py
git commit -m "feat(sdk): add with_open_actions() factory and propose_and_decide()"
```

---

## Phase D6: LLM-Backed Encoder and Proposer

### Task 7: LLM Encoder Backend

**Files:**
- Create: `backend/src/action_space/llm_encoder.py`
- Modify: `backend/src/llm/agent_runtime.py` (add `encode_action_modifiers` method)
- Test: `backend/tests/test_action_llm_encoder.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_llm_encoder.py
"""Tests for LLM-backed action encoding."""

import pytest

from src.action_space.llm_encoder import LLMEncoderBackend
from src.action_space.proposal import TextActionProposal, ToolActionProposal
from src.llm.schemas import ActionEncoding, LLMInvocationMetadata, LLMInvocationResult


class FakeLLMAdapter:
    """Returns canned ActionEncoding responses."""

    def __init__(self, modifiers: dict[str, float] | None = None) -> None:
        self._modifiers = modifiers or {"O": 0.5, "E": 0.3}

    def complete_json(self, *, system_prompt: str, user_prompt: str, response_model: type):
        encoding = ActionEncoding(
            modifiers=self._modifiers,
            confidence=0.8,
            rationale="test",
        )
        meta = LLMInvocationResult(
            raw_text="{}",
            metadata=LLMInvocationMetadata(model="test", provider="test"),
        )
        return encoding, meta


class TestLLMEncoderBackend:
    """LLM backend uses structured output for modifier estimation."""

    def test_encodes_tool_proposal(self) -> None:
        adapter = FakeLLMAdapter(modifiers={"O": 0.7, "C": 0.4})
        backend = LLMEncoderBackend(adapter=adapter)
        proposal = ToolActionProposal(
            name="search", description="search papers",
            tool_name="web_search", tool_args={"query": "test"},
        )
        modifiers = backend.estimate(proposal)
        assert modifiers["O"] == pytest.approx(0.7)
        assert modifiers["C"] == pytest.approx(0.4)

    def test_encodes_text_proposal(self) -> None:
        adapter = FakeLLMAdapter(modifiers={"A": 0.9})
        backend = LLMEncoderBackend(adapter=adapter)
        proposal = TextActionProposal(
            name="comfort", description="comfort someone",
            intent="comfort",
        )
        modifiers = backend.estimate(proposal)
        assert modifiers["A"] == pytest.approx(0.9)

    def test_graceful_degradation_on_error(self) -> None:

        class FailingAdapter:
            def complete_json(self, **kwargs):
                raise RuntimeError("LLM unavailable")

        backend = LLMEncoderBackend(adapter=FailingAdapter())
        proposal = ToolActionProposal(
            name="search", description="search",
            tool_name="web_search", tool_args={},
        )
        modifiers = backend.estimate(proposal)
        assert isinstance(modifiers, dict)
```

**Step 2: Run to verify failure**

Run: `cd backend && uv run pytest tests/test_action_llm_encoder.py -v`

**Step 3: Add encode_action_modifiers to AgentRuntime**

Add to `backend/src/llm/agent_runtime.py`:

```python
def encode_action_modifiers(
    self,
    *,
    name: str,
    description: str,
    kind: str,
    context: dict[str, Any],
) -> tuple[ActionEncoding, LLMInvocationResult]:
    """Estimate personality-dimension modifiers for an open-ended action."""
    system_prompt = (
        "You estimate how an action aligns with personality dimensions. "
        "Return JSON only. Keys: `modifiers` (dict[str, float] in [-1,1]), "
        "`confidence` (float 0-1), `rationale`. "
        "Personality dimensions: O(penness), C(onscientiousness), "
        "E(xtraversion), A(greeableness), N(euroticism), "
        "R(esilience), I(dealism), T(radition)."
    )
    user_prompt = json.dumps({
        "action_name": name,
        "action_description": description,
        "action_kind": kind,
        "context": context,
        "output_schema": "ActionEncoding",
    })
    return self._adapter.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=ActionEncoding,
    )
```

**Step 4: Implement llm_encoder.py**

```python
# backend/src/action_space/llm_encoder.py
"""LLM-backed action encoder backend.

Uses structured output to estimate personality-dimension modifiers.
Falls back to empty modifiers on LLM failure.
"""

from __future__ import annotations

from typing import Any

from src.action_space.proposal import _ProposalBase
from src.shared.logging import get_logger

_log = get_logger(module="action_space.llm_encoder")


class LLMEncoderBackend:
    """Encodes action proposals via LLM structured output."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Ask the LLM to estimate personality-dimension modifiers."""
        try:
            from src.llm.schemas import ActionEncoding

            encoding, _meta = self._adapter.complete_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(proposal),
                response_model=ActionEncoding,
            )
            return dict(encoding.modifiers)
        except Exception:
            _log.warning("llm_encoding_failed", action=proposal.name, exc_info=True)
            return {}

    def _system_prompt(self) -> str:
        return (
            "You estimate how an action aligns with personality dimensions. "
            "Return JSON only. Keys: `modifiers` (dict[str, float] in [-1,1]), "
            "`confidence` (float 0-1), `rationale`. "
            "Dimensions: O(penness), C(onscientiousness), E(xtraversion), "
            "A(greeableness), N(euroticism), R(esilience), I(dealism), T(radition)."
        )

    def _user_prompt(self, proposal: _ProposalBase) -> str:
        import json

        return json.dumps({
            "action_name": proposal.name,
            "action_description": proposal.description,
            "action_kind": getattr(proposal, "kind", "unknown"),
            "output_schema": "ActionEncoding",
        })
```

**Step 5: Run tests**

Run: `cd backend && uv run pytest tests/test_action_llm_encoder.py -v`
Expected: PASS

**Step 6: Run full suite + lint**

Run: `cd backend && uv run pytest -x -q && uv run ruff check src tests && uv run mypy src`

**Step 7: Commit**

```bash
git add src/action_space/llm_encoder.py src/llm/agent_runtime.py tests/test_action_llm_encoder.py
git commit -m "feat(action-space): add LLM-backed encoder with graceful degradation"
```

---

### Task 8: Caching Encoder Backend

**Files:**
- Create: `backend/src/action_space/caching_encoder.py`
- Test: `backend/tests/test_action_caching_encoder.py`

**Rationale:** LLM encoding is slow (~500ms per action). During a simulation with dynamic actions, the same tool or intent appears repeatedly across ticks. A caching layer memoizes LLM results keyed on `(kind, name, description)` to avoid redundant calls. The cache wraps any `EncoderBackend` via the decorator pattern — it doesn't know or care whether the inner backend is heuristic or LLM-backed.

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_caching_encoder.py
"""Tests for caching encoder backend decorator."""

import pytest

from src.action_space.caching_encoder import CachingEncoderBackend
from src.action_space.proposal import TextActionProposal, ToolActionProposal


class CountingBackend:
    """Tracks how many times estimate() is called."""

    def __init__(self, modifiers: dict[str, float] | None = None) -> None:
        self.call_count = 0
        self._modifiers = modifiers or {"O": 0.5}

    def estimate(self, proposal: object) -> dict[str, float]:
        """Return fixed modifiers and increment counter."""
        self.call_count += 1
        return dict(self._modifiers)


class TestCachingEncoderBackend:
    """CachingEncoderBackend memoizes inner backend calls."""

    def test_caches_identical_proposals(self) -> None:
        inner = CountingBackend(modifiers={"O": 0.7})
        cached = CachingEncoderBackend(inner=inner, max_size=128)
        proposal = ToolActionProposal(
            name="search",
            description="search papers",
            tool_name="web_search",
            tool_args={"query": "test"},
        )
        r1 = cached.estimate(proposal)
        r2 = cached.estimate(proposal)
        assert r1 == r2
        assert inner.call_count == 1

    def test_different_proposals_miss_cache(self) -> None:
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        p1 = ToolActionProposal(
            name="search", description="search papers",
            tool_name="web_search", tool_args={},
        )
        p2 = TextActionProposal(
            name="explain", description="explain concept",
            intent="explain",
        )
        cached.estimate(p1)
        cached.estimate(p2)
        assert inner.call_count == 2

    def test_same_name_different_description_misses(self) -> None:
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        p1 = TextActionProposal(
            name="explain", description="explain quantum physics",
            intent="explain",
        )
        p2 = TextActionProposal(
            name="explain", description="explain cooking",
            intent="explain",
        )
        cached.estimate(p1)
        cached.estimate(p2)
        assert inner.call_count == 2

    def test_evicts_when_full(self) -> None:
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner, max_size=2)
        proposals = [
            TextActionProposal(
                name=f"action_{i}", description=f"desc_{i}", intent="do"
            )
            for i in range(3)
        ]
        for p in proposals:
            cached.estimate(p)
        assert inner.call_count == 3
        # Re-request the first (evicted) — should miss
        cached.estimate(proposals[0])
        assert inner.call_count == 4
        # Re-request the third (still cached) — should hit
        cached.estimate(proposals[2])
        assert inner.call_count == 4

    def test_cache_stats(self) -> None:
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        proposal = ToolActionProposal(
            name="s", description="s",
            tool_name="t", tool_args={},
        )
        cached.estimate(proposal)
        cached.estimate(proposal)
        cached.estimate(proposal)
        stats = cached.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_clear_empties_cache(self) -> None:
        inner = CountingBackend()
        cached = CachingEncoderBackend(inner=inner)
        proposal = ToolActionProposal(
            name="s", description="s",
            tool_name="t", tool_args={},
        )
        cached.estimate(proposal)
        cached.clear()
        cached.estimate(proposal)
        assert inner.call_count == 2
```

**Step 2: Run tests to verify failure**

Run: `cd backend && uv run pytest tests/test_action_caching_encoder.py -v`
Expected: FAIL (module not found)

**Step 3: Implement caching_encoder.py**

```python
# backend/src/action_space/caching_encoder.py
"""Caching decorator for encoder backends.

Memoizes modifier estimation results keyed on (kind, name, description)
to avoid redundant LLM calls when the same action appears across ticks.
Uses an LRU eviction strategy with configurable max size.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from src.action_space.proposal import _ProposalBase
from src.shared.logging import get_logger

_log = get_logger(module="action_space.caching_encoder")


def _cache_key(proposal: _ProposalBase) -> tuple[str, str, str]:
    """Derive a stable cache key from proposal identity fields."""
    kind = getattr(proposal, "kind", "unknown")
    return (kind, proposal.name, proposal.description)


class CachingEncoderBackend:
    """LRU caching decorator over any EncoderBackend.

    Wraps an inner backend and caches results keyed on
    (kind, name, description). Thread-safety is not required
    because the simulator runs single-threaded per tick.
    """

    def __init__(
        self,
        inner: Any,
        max_size: int = 256,
    ) -> None:
        self._inner = inner
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str, str], dict[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def estimate(self, proposal: _ProposalBase) -> dict[str, float]:
        """Return cached modifiers or delegate to inner backend."""
        key = _cache_key(proposal)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return dict(self._cache[key])

        self._misses += 1
        modifiers = self._inner.estimate(proposal)
        self._cache[key] = dict(modifiers)

        if len(self._cache) > self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            _log.debug("cache_evicted", key=evicted_key)

        return dict(modifiers)

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss/size statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def clear(self) -> None:
        """Flush the cache and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
```

**Step 4: Run tests**

Run: `cd backend && uv run pytest tests/test_action_caching_encoder.py -v`
Expected: PASS

**Step 5: Run full suite + lint**

Run: `cd backend && uv run pytest -x -q && uv run ruff check src/action_space tests/test_action_caching_encoder.py && uv run mypy src/action_space`
Expected: All pass, no errors

**Step 6: Commit**

```bash
git add src/action_space/caching_encoder.py tests/test_action_caching_encoder.py
git commit -m "feat(action-space): add LRU caching decorator for encoder backends"
```

---

### Task 9: LLM Proposer Backend

**Files:**
- Create: `backend/src/action_space/llm_proposer.py`
- Test: `backend/tests/test_action_llm_proposer.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_action_llm_proposer.py
"""Tests for LLM-backed action proposer."""

from src.action_space.llm_proposer import LLMProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry
from src.llm.schemas import ActionSetProposal, LLMInvocationMetadata, LLMInvocationResult


class FakeLLMAdapter:
    """Returns canned ActionSetProposal responses."""

    def complete_json(self, *, system_prompt: str, user_prompt: str, response_model: type):
        proposal = ActionSetProposal(
            actions=[
                {"kind": "text", "name": "explain_concept", "description": "explain it",
                 "intent": "explain"},
                {"kind": "tool", "name": "search", "description": "search web",
                 "tool_name": "web_search", "tool_args": {"query": "test"}},
            ],
            rationale="context suggests information gathering",
        )
        meta = LLMInvocationResult(
            raw_text="{}",
            metadata=LLMInvocationMetadata(model="test", provider="test"),
        )
        return proposal, meta


class TestLLMProposerBackend:
    """LLM proposer generates context-aware candidate actions."""

    def test_proposes_actions_from_llm(self) -> None:
        adapter = FakeLLMAdapter()
        tool_reg = ToolRegistry()
        tool_reg.register(ToolCapability(
            name="web_search", description="search", parameter_schema={},
        ))
        backend = LLMProposerBackend(adapter=adapter, tool_registry=tool_reg)
        proposals = backend.propose(
            state={"energy": 0.8, "mood": 0.3},
            trajectory=[],
            goals=["understand the topic"],
        )
        assert len(proposals) >= 1
        names = [p.name for p in proposals]
        assert "explain_concept" in names or "search" in names

    def test_graceful_degradation_returns_empty(self) -> None:
        class FailingAdapter:
            def complete_json(self, **kwargs):
                raise RuntimeError("LLM down")

        backend = LLMProposerBackend(adapter=FailingAdapter())
        proposals = backend.propose(state={}, trajectory=[], goals=[])
        assert proposals == []
```

**Step 2 through 6:** Standard TDD cycle — fail, implement, pass, lint, commit.

**Implementation** in `backend/src/action_space/llm_proposer.py`:

```python
# backend/src/action_space/llm_proposer.py
"""LLM-backed action proposer backend.

Asks the LLM to generate contextually appropriate action candidates,
then parses them into typed ActionProposal objects.
"""

from __future__ import annotations

from typing import Any

from src.action_space.proposal import (
    ClassicActionProposal,
    TextActionProposal,
    ToolActionProposal,
    _ProposalBase,
)
from src.action_space.registry import ToolRegistry
from src.shared.logging import get_logger

_log = get_logger(module="action_space.llm_proposer")


class LLMProposerBackend:
    """Proposes actions via LLM structured output."""

    def __init__(
        self,
        adapter: Any,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._adapter = adapter
        self._tools = tool_registry

    def propose(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> list[_ProposalBase]:
        """Ask the LLM to propose candidate actions."""
        try:
            from src.llm.schemas import ActionSetProposal

            result, _meta = self._adapter.complete_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(state, trajectory, goals),
                response_model=ActionSetProposal,
            )
            return self._parse_actions(result.actions)
        except Exception:
            _log.warning("llm_proposal_failed", exc_info=True)
            return []

    def _system_prompt(self) -> str:
        tools_ctx = ""
        if self._tools:
            tools_ctx = f"\nAvailable tools:\n{self._tools.to_prompt_context()}\n"
        return (
            "You propose candidate actions for a personality-driven agent. "
            "Return JSON with keys: `actions` (list of action objects), `rationale`. "
            "Each action object must have: `kind` (tool|text|classic), `name`, `description`. "
            "Tool actions need: `tool_name`, `tool_args`. "
            "Text actions need: `intent`. "
            "Classic actions need: `modifiers` (dict[str, float] in [-1,1])."
            f"{tools_ctx}"
        )

    def _user_prompt(
        self,
        state: dict[str, Any],
        trajectory: list[dict[str, Any]],
        goals: list[str],
    ) -> str:
        import json

        return json.dumps({
            "current_state": state,
            "recent_trajectory": trajectory[-5:] if trajectory else [],
            "goals": goals,
            "output_schema": "ActionSetProposal",
        })

    def _parse_actions(self, raw: list[dict[str, Any]]) -> list[_ProposalBase]:
        proposals: list[_ProposalBase] = []
        for item in raw:
            try:
                kind = item.get("kind", "classic")
                if kind == "tool":
                    proposals.append(ToolActionProposal(**item))
                elif kind == "text":
                    proposals.append(TextActionProposal(**item))
                else:
                    proposals.append(ClassicActionProposal(**item))
            except Exception:
                _log.debug("skipping_invalid_proposal", item=item, exc_info=True)
        return proposals
```

**Commit:**

```bash
git add src/action_space/llm_proposer.py tests/test_action_llm_proposer.py
git commit -m "feat(action-space): add LLM-backed proposer backend"
```

---

## Phase D7: Simulator Integration

### Task 10: Dynamic Action Space in TemporalSimulator

**Files:**
- Modify: `backend/src/temporal/simulator.py` (add optional proposer/encoder to tick)
- Create: `backend/tests/test_simulator_open_ended.py`

This is the most delicate task — it must not break the existing tick loop while enabling dynamic actions.

**Design:** Add an optional `action_pipeline` parameter to `TemporalSimulator.__init__()`. When present, `tick()` calls `pipeline.propose_and_encode()` to get actions dynamically instead of using `self.actions`. When absent, behavior is identical to today.

**Step 1: Write the failing tests**

```python
# backend/tests/test_simulator_open_ended.py
"""Tests for dynamic action space in TemporalSimulator."""

from src.action_space.encoder import ActionEncoder, HeuristicEncoderBackend
from src.action_space.proposal import ClassicActionProposal
from src.action_space.proposer import ActionProposer, StaticProposerBackend
from src.personality.dimensions import DimensionRegistry
from src.sdk import AgentSDK


class TestSimulatorWithDynamicActions:
    """TemporalSimulator works with dynamic action proposals."""

    def test_tick_with_dynamic_actions(self) -> None:
        classics = [
            ClassicActionProposal(name="bold", description="bold", modifiers={"O": 1.0, "R": 0.8}),
            ClassicActionProposal(name="safe", description="safe", modifiers={"C": 0.9, "T": 0.8}),
        ]
        backend = StaticProposerBackend(defaults=classics)
        sdk = AgentSDK.with_open_actions(proposer_backend=backend, include_withdraw=True)
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="pitch")
        sim = sdk.simulator(personality, actions=[])
        trace = sim.run([scenario], outcomes=[0.4])
        assert trace.ticks[0].chosen_action in ("bold", "safe", "Withdraw")

    def test_static_actions_still_work(self) -> None:
        sdk = AgentSDK.default()
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        actions = [sdk.action("bold", {"O": 1.0}), sdk.action("safe", {"C": 0.9})]
        scenario = sdk.scenario({"O": 0.9}, name="test")
        sim = sdk.simulator(personality, actions)
        trace = sim.run([scenario], outcomes=[0.4])
        assert trace.ticks[0].chosen_action in ("bold", "safe")
```

**Step 2 through 6:** TDD cycle.

**Implementation strategy:** Add an `ActionPipeline` dataclass that wraps proposer + encoder. Pass it optionally to the simulator. In `tick()`, if pipeline exists and `self.actions` is empty, use `pipeline.propose_and_encode()`.

**Commit:**

```bash
git add src/temporal/simulator.py src/action_space/pipeline.py tests/test_simulator_open_ended.py
git commit -m "feat(temporal): wire dynamic action pipeline into simulator tick loop"
```

---

## Phase D8: API and Persistence

### Task 11: API Endpoints for Open-Ended Decisions

**Files:**
- Modify: `backend/src/api/schemas.py` (add `OpenEndedDecisionRequest`)
- Modify: `backend/src/api/simulation_router.py` (add `POST /decide/open`)
- Test: `backend/tests/test_api_open_ended.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_api_open_ended.py
"""Tests for open-ended decision API endpoints."""

from tests.api_support import make_test_client


class TestOpenEndedDecideEndpoint:
    """POST /decide/open with action proposals."""

    def test_classic_proposals(self) -> None:
        client = make_test_client()
        resp = client.post("/decide/open", json={
            "personality": {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7,
                            "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2},
            "scenario": {"name": "test", "values": {"O": 0.9}},
            "proposals": [
                {"kind": "classic", "name": "bold", "description": "bold",
                 "modifiers": {"O": 1.0}},
                {"kind": "classic", "name": "safe", "description": "safe",
                 "modifiers": {"C": 0.9}},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["chosen_action"] in ("bold", "safe")
        assert "proposals" in data

    def test_mixed_proposals(self) -> None:
        client = make_test_client()
        resp = client.post("/decide/open", json={
            "personality": {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7,
                            "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2},
            "scenario": {"name": "test", "values": {"O": 0.9}},
            "proposals": [
                {"kind": "classic", "name": "bold", "description": "bold",
                 "modifiers": {"O": 1.0}},
                {"kind": "text", "name": "explain", "description": "explain",
                 "intent": "explain"},
            ],
        })
        assert resp.status_code == 200
```

**Step 2 through 6:** TDD cycle.

**Commit:**

```bash
git add src/api/schemas.py src/api/simulation_router.py tests/test_api_open_ended.py
git commit -m "feat(api): add POST /decide/open for open-ended action proposals"
```

---

### Task 12: Persist Proposals in RunStore

**Files:**
- Modify: `backend/src/api/run_schemas.py` (add proposals to RunConfig)
- Modify: `backend/src/api/run_store.py` (store proposal metadata alongside tick events)
- Test: `backend/tests/test_run_store_proposals.py`

This task extends the tick event payload to include proposal metadata (kind, tool_name, intent, etc.) so open-ended decisions are fully reconstructible from persistence.

**Commit:**

```bash
git add src/api/run_schemas.py src/api/run_store.py tests/test_run_store_proposals.py
git commit -m "feat(persistence): store action proposal metadata in tick events"
```

---

## Phase D9: Full Integration Test

### Task 13: End-to-End Open-Ended Action Pipeline

**Files:**
- Create: `backend/tests/test_open_ended_integration.py`

```python
# backend/tests/test_open_ended_integration.py
"""Integration test: full open-ended action pipeline."""

from src.action_space.encoder import ActionEncoder, HeuristicEncoderBackend
from src.action_space.executor import ActionExecutor, NoopToolHandler
from src.action_space.proposal import ClassicActionProposal, ToolActionProposal
from src.action_space.proposer import ActionProposer, StaticProposerBackend
from src.action_space.registry import ToolCapability, ToolRegistry
from src.sdk import AgentSDK


class TestFullPipeline:
    """End-to-end: propose → encode → decide → execute."""

    def test_tool_plus_classic_pipeline(self) -> None:
        tool_reg = ToolRegistry()
        tool_reg.register(ToolCapability(
            name="web_search",
            description="search the web",
            parameter_schema={"query": {"type": "string"}},
            personality_hint={"O": 0.8, "E": 0.3},
        ))

        classics = [
            ClassicActionProposal(name="safe", description="safe", modifiers={"C": 0.9, "T": 0.8}),
        ]
        backend = StaticProposerBackend(defaults=classics)

        sdk = AgentSDK.with_open_actions(
            proposer_backend=backend,
            tool_registry=tool_reg,
            include_withdraw=True,
        )

        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        scenario = sdk.scenario({"O": 0.9, "N": 0.7}, name="research_meeting")

        result = sdk.propose_and_decide(personality, scenario)
        assert result.chosen_action in ("use_web_search", "safe", "Withdraw")
        assert len(result.proposals) == 3

    def test_backward_compat_full_simulation(self) -> None:
        sdk = AgentSDK.default()
        personality = sdk.personality(
            {"O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7, "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2}
        )
        actions = [sdk.action("bold", {"O": 1.0}), sdk.action("safe", {"C": 0.9})]
        scenarios = [sdk.scenario({"O": 0.9}, name="s1"), sdk.scenario({"C": 0.8}, name="s2")]
        trace = sdk.simulator(personality, actions).run(scenarios, outcomes=[0.4, 0.2])
        assert len(trace.ticks) == 2
```

**Commit:**

```bash
git add tests/test_open_ended_integration.py
git commit -m "test: add end-to-end integration test for open-ended action pipeline"
```

---

## Summary

| Phase | Tasks | New Files | Modified Files |
|-------|-------|-----------|----------------|
| D1: Types & Registry | 1-2 | 4 | 0 |
| D2: Encoder | 3 | 3 | 3 |
| D3: Proposer | 4 | 2 | 1 |
| D4: Executor | 5 | 1 | 0 |
| D5: SDK Integration | 6 | 1 | 2 |
| D6: LLM Backends + Cache | 7-9 | 3 | 1 |
| D7: Simulator | 10 | 2 | 1 |
| D8: API & Persistence | 11-12 | 2 | 3 |
| D9: Integration | 13 | 1 | 0 |

**Total: 13 tasks, 19 new files, 11 modified files**

**Key invariant:** All 524 existing tests pass at every commit. The open-ended action space is additive — it never changes the behavior of `AgentSDK.default()` or `AgentSDK.decide()`.

**Architectural result:** The action space becomes a pluggable pipeline (propose → encode → decide → execute) where:
- **Propose** generates candidates from LLM + tools + context
- **Encode** maps any action to the personality-dimension modifier space (with LRU cache to amortize LLM cost)
- **Decide** uses the unchanged Boltzmann/EFE/self-evidencing machinery
- **Execute** dispatches tool calls, API requests, or text generation

## Future Extension: Embedding-Based Encoder

The `EncoderBackend` protocol accepts a fourth implementation strategy not covered in this plan: a **learned embedding model** that maps action descriptions to modifier vectors without LLM round-trips. This becomes viable once enough LLM encoding data accumulates in the cache or persistence layer to train a small regression model (action embedding → 8D modifier vector).

The integration path is trivial:

1. Export cached `(description, modifiers)` pairs from `CachingEncoderBackend` or `RunStore`
2. Train a sentence-transformer → linear head model
3. Wrap it as `EmbeddingEncoderBackend` implementing `EncoderBackend.estimate()`
4. Compose: `CachingEncoderBackend(inner=EmbeddingEncoderBackend(model=...))` with LLM fallback for out-of-distribution actions

This is intentionally deferred — the LLM + cache path is sufficient for the current scale and avoids a training data cold-start problem.
