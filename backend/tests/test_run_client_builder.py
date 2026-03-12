"""Tests for RunClientBuilder SDK mode resolution."""

from src.api.run_client_builder import RunClientBuilder
from src.sdk.builders import build_registry, build_scenario


def _balanced() -> dict[str, float]:
    return {k: 0.5 for k in "OCEANRIT"}


def test_builder_uses_efe_sdk_when_config_specifies() -> None:
    """RunClientBuilder should use EFE SDK when config has sdk_mode='efe'."""
    config = {
        "personality": _balanced(),
        "actions": [{"name": "Act", "modifiers": {"O": 0.3}}],
        "sdk_mode": "efe",
    }
    builder = RunClientBuilder()
    client = builder.build(config, prior_ticks=[])
    scenario = build_scenario(_balanced(), build_registry(), name="t")
    rec = client.tick(scenario)
    assert rec.tick == 0


def test_builder_uses_self_evidencing_sdk_when_config_specifies() -> None:
    """RunClientBuilder uses self-evidencing SDK for sdk_mode='self_evidencing'."""
    config = {
        "personality": _balanced(),
        "actions": [{"name": "Act", "modifiers": {"O": 0.3}}],
        "self_model": _balanced(),
        "sdk_mode": "self_evidencing",
    }
    builder = RunClientBuilder()
    client = builder.build(config, prior_ticks=[])
    scenario = build_scenario(_balanced(), build_registry(), name="t")
    rec = client.tick(scenario)
    assert rec.tick == 0


def test_builder_defaults_to_precision_sdk() -> None:
    """RunClientBuilder defaults to precision SDK when no sdk_mode."""
    config = {
        "personality": _balanced(),
        "actions": [{"name": "Act", "modifiers": {"O": 0.3}}],
    }
    builder = RunClientBuilder()
    client = builder.build(config, prior_ticks=[])
    scenario = build_scenario(_balanced(), build_registry(), name="t")
    rec = client.tick(scenario)
    assert rec.tick == 0
