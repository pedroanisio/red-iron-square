"""Helpers for mapping demo domain state into API-facing payloads.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from src.demo.models import (
    DemoAgentSnapshot,
    DemoPersona,
    DemoScenario,
    DemoSessionState,
)
from src.demo.personas import DEFAULT_PERSONAS, DISPLAY_EMOTION_LABELS
from src.demo.schemas import DemoAgentResponse, DemoSessionResponse

DEFAULT_ACTIONS = [
    {
        "name": "explore",
        "modifiers": {"O": 0.9, "E": 0.4, "R": 0.7, "T": -0.3},
        "description": "Lean into novelty and possibility.",
    },
    {
        "name": "stabilize",
        "modifiers": {"C": 0.7, "A": 0.5, "T": 0.8, "N": -0.2},
        "description": "Preserve continuity and reduce risk.",
    },
]


def build_demo_personas(swapped: bool = False) -> dict[str, DemoPersona]:
    """Return persona definitions, optionally swapping trait vectors."""
    if not swapped:
        return DEFAULT_PERSONAS
    return {
        "luna": DemoPersona(
            key="luna",
            name="Luna",
            summary=DEFAULT_PERSONAS["luna"].summary,
            traits=DEFAULT_PERSONAS["marco"].traits,
        ),
        "marco": DemoPersona(
            key="marco",
            name="Marco",
            summary=DEFAULT_PERSONAS["marco"].summary,
            traits=DEFAULT_PERSONAS["luna"].traits,
        ),
    }


def build_run_config(persona: DemoPersona) -> dict[str, Any]:
    """Build a self-aware run config for one demo persona."""
    seed = hash(persona.key) % (2**31)
    return {
        "personality": persona.traits,
        "actions": DEFAULT_ACTIONS,
        "temperature": 0.8,
        "self_model": persona.traits,
        "seed": seed,
    }


def derive_initial_affect(traits: dict[str, float]) -> dict[str, float]:
    """Compute personality-grounded initial mood, energy, and calm.

    Extraversion (E) lifts baseline energy, Neuroticism (N) lowers
    initial mood, and Resilience (R) raises initial calm.
    """
    extraversion = traits.get("E", 0.5)
    neuroticism = traits.get("N", 0.5)
    resilience = traits.get("R", 0.5)
    mood = round(0.2 * (resilience - neuroticism), 2)
    energy = round(0.4 + 0.2 * extraversion, 2)
    calm = round(0.4 + 0.2 * resilience, 2)
    return {"mood": mood, "energy": energy, "calm": calm}


def build_initial_agents(
    personas: dict[str, DemoPersona],
) -> dict[str, DemoAgentSnapshot]:
    """Construct personality-grounded agent snapshots for a new session."""
    agents: dict[str, DemoAgentSnapshot] = {}
    for key, persona in personas.items():
        affect = derive_initial_affect(persona.traits)
        agents[key] = DemoAgentSnapshot(
            key=persona.key,
            name=persona.name,
            summary=persona.summary,
            traits=persona.traits,
            mood=affect["mood"],
            energy=affect["energy"],
            calm=affect["calm"],
        )
    return agents


def session_to_response(session: DemoSessionState) -> DemoSessionResponse:
    """Convert internal session state to API-safe schema."""
    return DemoSessionResponse(
        session_id=session.session_id,
        act_number=session.act_number,
        turn_count=session.turn_count,
        agents=[
            DemoAgentResponse(
                key=agent.key,
                name=agent.name,
                summary=agent.summary,
                mood=agent.mood,
                energy=agent.energy,
                calm=agent.calm,
                emotion_label=agent.emotion_label,
            )
            for agent in session.agents.values()
        ],
    )


def update_snapshot_from_tick(
    snapshot: DemoAgentSnapshot,
    tick: dict[str, Any],
    transcript: str,
    emotion_label: str | None = None,
) -> DemoAgentSnapshot:
    """Apply one tick payload to a frontend-facing agent snapshot."""
    state_after = tick["state_after"]
    arousal = float(state_after.get("arousal", 0.5))
    frustration = float(state_after.get("frustration", 0.5))
    calm = max(0.0, min(1.0, 1.0 - ((arousal + frustration) / 2.0)))
    display_label = emotion_label or _resolve_emotion_label(tick)
    snapshot.mood = float(state_after.get("mood", 0.0))
    snapshot.energy = float(state_after.get("energy", 0.5))
    snapshot.calm = calm
    snapshot.emotion_label = display_label
    snapshot.transcript.append(transcript)
    return snapshot


def modulate_outcome(
    base_outcome: float,
    persona: DemoPersona,
    scenario_values: dict[str, float],
) -> float:
    """Shift a scripted outcome based on persona-scenario alignment.

    Computes how well the persona's traits align with what the scenario
    demands, then nudges the base outcome accordingly.  High-N personas
    experience negative events more intensely; high-R personas dampen them.
    """
    shared_keys = set(persona.traits) & set(scenario_values)
    if not shared_keys:
        return base_outcome
    alignment = sum(persona.traits[k] * scenario_values[k] for k in shared_keys) / len(
        shared_keys
    )
    neuroticism = persona.traits.get("N", 0.5)
    resilience = persona.traits.get("R", 0.5)
    shift = 0.3 * (alignment - 0.5) + 0.15 * (resilience - neuroticism)
    modulated = base_outcome + shift
    return float(max(-1.0, min(1.0, modulated)))


def build_custom_scenario(text: str) -> DemoScenario:
    """Create a neutral fallback scenario from audience free text."""
    return DemoScenario(
        key="custom",
        name="Audience Scenario",
        description=text,
        values={key: 0.5 for key in DEFAULT_PERSONAS["luna"].traits},
        forced_outcome=None,
    )


def _resolve_emotion_label(tick: dict[str, Any]) -> str:
    readings = tick.get("self_emotions") or tick.get("emotions") or []
    if not readings:
        return "Neutral"
    strongest = max(readings, key=lambda reading: float(reading.get("intensity", 0.0)))
    label = str(strongest.get("label", "Neutral"))
    return DISPLAY_EMOTION_LABELS.get(label, label.replace("_", " ").title())
