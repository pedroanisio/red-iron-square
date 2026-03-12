"""Tests for the SDK facade layer."""

import json

import numpy as np
import pytest
from src.sdk import AgentSDK


class TestAgentSDK:
    """Tests for the high-level SDK API."""

    def setup_method(self) -> None:
        self.sdk = AgentSDK.default()
        self.personality = self.sdk.personality(
            {
                "O": 0.8,
                "C": 0.5,
                "E": 0.3,
                "A": 0.7,
                "N": 0.4,
                "R": 0.9,
                "I": 0.6,
                "T": 0.2,
            }
        )
        self.scenario = self.sdk.scenario(
            {"O": 0.9, "N": 0.7},
            name="pitch_meeting",
        )
        self.actions = [
            self.sdk.action("bold", {"O": 1.0, "R": 0.8, "N": -0.3}),
            self.sdk.action("safe", {"C": 0.9, "T": 0.8}),
        ]

    def test_sparse_builders_fill_missing_values(self) -> None:
        assert self.personality["O"] == pytest.approx(0.8)
        assert self.personality["T"] == pytest.approx(0.2)
        assert self.scenario["C"] == pytest.approx(0.0)

    def test_action_rejects_out_of_range_modifier(self) -> None:
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            self.sdk.action("bad", {"O": 1.5})

    def test_decide_returns_json_safe_result(self) -> None:
        result = self.sdk.decide(
            self.personality,
            self.scenario,
            self.actions,
            rng=np.random.default_rng(42),
        )
        payload = result.model_dump()
        json.dumps(payload)
        assert result.chosen_action in {"bold", "safe"}
        assert payload["probabilities"]["bold"] + payload["probabilities"][
            "safe"
        ] == pytest.approx(1.0)
        assert set(payload["activations"].keys()) == set(self.sdk.registry.keys)

    def test_temporal_simulator_run_returns_trace(self) -> None:
        simulator = self.sdk.simulator(
            self.personality,
            self.actions,
            rng=np.random.default_rng(42),
        )
        trace = simulator.run([self.scenario], outcomes=[0.4])
        payload = trace.model_dump()
        json.dumps(payload)
        assert len(trace.ticks) == 1
        assert trace.ticks[0].scenario["name"] == "pitch_meeting"
        assert trace.ticks[0].state_after["energy"] <= 1.0

    def test_self_aware_simulator_run_returns_trace(self) -> None:
        simulator = self.sdk.self_aware_simulator(
            self.personality,
            self.sdk.initial_self_model(
                {
                    "O": 0.7,
                    "C": 0.5,
                    "E": 0.4,
                    "A": 0.6,
                    "N": 0.4,
                    "R": 0.8,
                    "I": 0.6,
                    "T": 0.3,
                }
            ),
            self.actions,
            rng=np.random.default_rng(42),
        )
        trace = simulator.run([self.scenario], outcomes=[0.6])
        payload = trace.model_dump()
        json.dumps(payload)
        assert len(trace.ticks) == 1
        assert set(trace.ticks[0].psi_hat.keys()) == set(self.sdk.registry.keys)
        assert 0.0 <= trace.ticks[0].prediction_error <= 1.0


class TestSDKDecideEFE:
    """AgentSDK.decide() should use EFE engine when EFE mode active."""

    def test_decide_uses_efe_engine(self) -> None:
        sdk = AgentSDK.with_efe()
        personality = sdk.personality({k: 0.5 for k in "OCEANRIT"})
        scenario = sdk.scenario({k: 0.5 for k in "OCEANRIT"}, name="test")
        actions = [
            sdk.action("Act", {"O": 0.3, "E": 0.2}),
            sdk.action("Wait", {"O": -0.1}),
        ]
        result = sdk.decide(personality, scenario, actions)
        assert result.chosen_action in {"Act", "Wait"}
        assert len(result.probabilities) == 2

    def test_decide_default_uses_base_engine(self) -> None:
        sdk = AgentSDK.default()
        personality = sdk.personality({k: 0.5 for k in "OCEANRIT"})
        scenario = sdk.scenario({k: 0.5 for k in "OCEANRIT"}, name="test")
        actions = [sdk.action("Act", {"O": 0.3})]
        result = sdk.decide(personality, scenario, actions)
        assert result.chosen_action == "Act"
