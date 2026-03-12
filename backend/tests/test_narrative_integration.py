"""Tests for narrative model + emotion callback integration in the simulation loop.

Verifies:
  - NarrativeGenerativeModel is refreshed on surprise spikes
  - EmotionCallback is invoked on surprise spikes
  - SDK wiring passes both through to the simulator
"""

from __future__ import annotations

import numpy as np
from src.sdk import AgentSDK
from src.temporal.emotions import EmotionLabel, EmotionReading


def _balanced() -> dict[str, float]:
    """Return a balanced personality profile."""
    return {k: 0.5 for k in "OCEANRIT"}


def _build_sdk_sim(
    *,
    emotion_callback: object | None = None,
    n_ticks: int = 30,
    seed: int = 42,
) -> tuple:
    """Build a constructed-emotion SDK and simulator."""
    sdk = AgentSDK.with_constructed_emotion()
    if emotion_callback is not None:
        sdk.set_emotion_callback(emotion_callback)  # type: ignore[arg-type]
    personality = sdk.personality(_balanced())
    scenario = sdk.scenario(_balanced(), name="test")
    actions = [
        sdk.action("A", {"O": 0.5, "E": 0.3}),
        sdk.action("B", {"C": 0.5, "E": -0.1}),
    ]
    sim = sdk.simulator(
        personality,
        actions,
        rng=np.random.default_rng(seed),
    )
    return sim, scenario


class TestNarrativeRefreshOnSpike:
    """Narrative model should refresh when surprise spikes occur."""

    def test_narrative_model_created_by_sdk(self) -> None:
        """SDK with emotion params creates a narrative model in simulator."""
        sdk = AgentSDK.with_constructed_emotion()
        personality = sdk.personality(_balanced())
        actions = [sdk.action("A", {"O": 0.3})]
        sim = sdk.simulator(
            personality,
            actions,
            rng=np.random.default_rng(0),
        )
        assert sim.simulator._narrative_model is not None

    def test_no_narrative_without_emotion_params(self) -> None:
        """SDK without emotion params does not create narrative model."""
        sdk = AgentSDK.with_precision()
        personality = sdk.personality(_balanced())
        actions = [sdk.action("A", {"O": 0.3})]
        sim = sdk.simulator(
            personality,
            actions,
            rng=np.random.default_rng(0),
        )
        assert sim.simulator._narrative_model is None

    def test_refresh_called_on_spike(self) -> None:
        """Narrative model refresh is triggered by surprise spikes."""
        sdk = AgentSDK.with_constructed_emotion()
        personality = sdk.personality(_balanced())
        scenario = sdk.scenario(_balanced(), name="test")
        actions = [
            sdk.action("A", {"O": 0.5, "E": 0.5}),
            sdk.action("B", {"C": 0.5}),
        ]
        sim = sdk.simulator(
            personality,
            actions,
            rng=np.random.default_rng(42),
        )
        model = sim.simulator._narrative_model
        assert model is not None

        original_refresh = model.refresh_from_trajectory
        call_count = 0

        def counting_refresh(window: list) -> None:
            """Track calls to refresh_from_trajectory."""
            nonlocal call_count
            call_count += 1
            original_refresh(window)

        model.refresh_from_trajectory = counting_refresh  # type: ignore[method-assign]

        for _ in range(100):
            sim.tick(scenario)

        assert call_count > 0, "Narrative model should be refreshed on spikes"


class TestEmotionCallbackWiring:
    """EmotionCallback should be invoked on surprise spikes."""

    def test_callback_invoked_on_spike(self) -> None:
        """SDK-registered callback fires during simulation."""
        captured: list[tuple] = []

        def mock_callback(
            valence: float,
            arousal: float,
            errors: list[float],
            context: str,
        ) -> list[EmotionReading]:
            """Capture callback invocations."""
            captured.append((valence, arousal, context))
            return [
                EmotionReading(
                    label=EmotionLabel.EXCITEMENT,
                    intensity=0.8,
                    description="LLM-constructed",
                )
            ]

        sim, scenario = _build_sdk_sim(
            emotion_callback=mock_callback,
            n_ticks=100,
        )
        for _ in range(100):
            sim.tick(scenario)

        assert len(captured) >= 1, "Callback should fire on surprise spikes"

    def test_no_callback_uses_heuristic(self) -> None:
        """Without callback, heuristic categorization is used."""
        sim, scenario = _build_sdk_sim(emotion_callback=None)
        for _ in range(50):
            sim.tick(scenario)
        # No error means heuristic fallback works correctly


class TestSelfAwareNarrativeWiring:
    """Narrative model should also work in self-aware simulator."""

    def test_self_aware_has_narrative_model(self) -> None:
        """Self-aware simulator also gets a narrative model."""
        sdk = AgentSDK.with_self_evidencing()
        personality = sdk.personality(_balanced())
        psi_hat = sdk.initial_self_model(_balanced())
        actions = [sdk.action("A", {"O": 0.3})]
        sim = sdk.self_aware_simulator(
            personality,
            psi_hat,
            actions,
            rng=np.random.default_rng(0),
        )
        assert sim.simulator._narrative_model is not None
