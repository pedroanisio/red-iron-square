"""Tests for ElevenLabs audio integration in the Two Minds demo.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.demo.audio import (
    AudioResult,
    AudioTagInjector,
    ElevenLabsConfigError,
    ElevenLabsProvider,
    VoiceProfile,
    VoiceSettings,
    VoiceSettingsCalculator,
)

# ── Factories ─────────────────────────────────────────────────────


def _agent_state(
    mood: float = 0.0,
    energy: float = 0.5,
    arousal: float = 0.3,
    frustration: float = 0.2,
    prev_mood: float | None = None,
    identity_drift: float = 0.0,
) -> dict[str, float]:
    """Build a minimal agent state dict for testing."""
    state: dict[str, float] = {
        "mood": mood,
        "energy": energy,
        "arousal": arousal,
        "frustration": frustration,
        "identity_drift": identity_drift,
    }
    if prev_mood is not None:
        state["prev_mood"] = prev_mood
    return state


def _luna_profile() -> VoiceProfile:
    return VoiceProfile(
        voice_id="luna_voice_id",
        agent_key="luna",
        base_stability=0.50,
        base_speed=0.97,
        allowed_tags=frozenset(
            ["sighs", "softly", "hesitantly", "voice breaking", "pauses"]
        ),
    )


def _marco_profile() -> VoiceProfile:
    return VoiceProfile(
        voice_id="marco_voice_id",
        agent_key="marco",
        base_stability=0.32,
        base_speed=1.02,
        allowed_tags=frozenset(["excited", "laughs", "with confidence", "frustrated"]),
    )


# ── VoiceSettingsCalculator ───────────────────────────────────────


class TestVoiceSettingsCalculator:
    """Verify state-to-voice-settings mapping."""

    def test_neutral_state_returns_baseline(self) -> None:
        """Neutral agent state yields settings near profile baselines."""
        calc = VoiceSettingsCalculator()
        settings = calc.compute(_agent_state(), _luna_profile(), act_number=1)

        assert isinstance(settings, VoiceSettings)
        assert 0.40 <= settings.stability <= 0.55
        assert settings.similarity_boost == 0.75
        assert 0.0 <= settings.style <= 0.15
        assert 0.92 <= settings.speed <= 1.02

    def test_high_emotion_lowers_stability(self) -> None:
        """High arousal + strong mood push stability down."""
        calc = VoiceSettingsCalculator()
        neutral = calc.compute(
            _agent_state(mood=0.0, arousal=0.2),
            _marco_profile(),
            act_number=1,
        )
        emotional = calc.compute(
            _agent_state(mood=-0.8, arousal=0.9),
            _marco_profile(),
            act_number=1,
        )

        assert emotional.stability < neutral.stability

    def test_high_energy_increases_speed(self) -> None:
        """High energy nudges speaking rate upward."""
        calc = VoiceSettingsCalculator()
        low = calc.compute(_agent_state(energy=0.2), _luna_profile(), act_number=1)
        high = calc.compute(_agent_state(energy=0.9), _luna_profile(), act_number=1)

        assert high.speed > low.speed

    def test_strong_mood_increases_style(self) -> None:
        """Extreme mood (positive or negative) raises style."""
        calc = VoiceSettingsCalculator()
        neutral = calc.compute(_agent_state(mood=0.0), _luna_profile(), act_number=1)
        negative = calc.compute(_agent_state(mood=-0.8), _luna_profile(), act_number=1)
        positive = calc.compute(_agent_state(mood=0.7), _luna_profile(), act_number=1)

        assert negative.style > neutral.style
        assert positive.style > neutral.style

    def test_similarity_boost_always_fixed(self) -> None:
        """Similarity boost stays constant across all states."""
        calc = VoiceSettingsCalculator()
        for mood in (-0.9, 0.0, 0.9):
            settings = calc.compute(
                _agent_state(mood=mood), _luna_profile(), act_number=1
            )
            assert settings.similarity_boost == 0.75

    def test_act3_disables_speaker_boost(self) -> None:
        """Act 3 favors latency — speaker boost off."""
        calc = VoiceSettingsCalculator()
        act1 = calc.compute(_agent_state(), _luna_profile(), act_number=1)
        act3 = calc.compute(_agent_state(), _luna_profile(), act_number=3)

        assert act1.use_speaker_boost is True
        assert act3.use_speaker_boost is False

    def test_settings_stay_within_api_bounds(self) -> None:
        """All extreme states must produce API-valid ranges."""
        calc = VoiceSettingsCalculator()
        extremes = [
            _agent_state(mood=-1.0, energy=0.0, arousal=1.0),
            _agent_state(mood=1.0, energy=1.0, arousal=1.0),
        ]
        for state in extremes:
            settings = calc.compute(state, _marco_profile(), act_number=1)
            assert 0.0 <= settings.stability <= 1.0
            assert 0.0 <= settings.similarity_boost <= 1.0
            assert 0.0 <= settings.style <= 1.0
            assert 0.5 <= settings.speed <= 2.0


# ── AudioTagInjector ──────────────────────────────────────────────


class TestAudioTagInjector:
    """Verify state-driven audio tag injection and stripping."""

    def test_low_mood_low_energy_injects_sigh(self) -> None:
        """Drained agent triggers a sigh tag."""
        injector = AudioTagInjector()
        result = injector.inject(
            "I feel exhausted.",
            _agent_state(mood=-0.6, energy=0.2),
            _luna_profile(),
        )

        assert result.startswith("[")
        assert "I feel exhausted." in result

    def test_high_mood_high_arousal_injects_excitement(self) -> None:
        """Excited agent triggers an excitement tag."""
        injector = AudioTagInjector()
        result = injector.inject(
            "This is amazing!",
            _agent_state(mood=0.7, arousal=0.8),
            _marco_profile(),
        )

        assert "[excited]" in result or "[with confidence]" in result

    def test_identity_drift_injects_hesitation(self) -> None:
        """High identity drift triggers hesitation."""
        injector = AudioTagInjector()
        result = injector.inject(
            "I'm not sure who I am anymore.",
            _agent_state(identity_drift=0.5),
            _luna_profile(),
        )

        assert "[hesitantly]" in result

    def test_rapid_mood_drop_injects_voice_breaking(self) -> None:
        """Sharp mood decline triggers voice breaking."""
        injector = AudioTagInjector()
        result = injector.inject(
            "Everything just fell apart.",
            _agent_state(mood=-0.5, prev_mood=0.1),
            _luna_profile(),
        )

        assert "[voice breaking]" in result or "[pauses]" in result

    def test_max_two_tags_injected(self) -> None:
        """Never inject more than two audio tags per response."""
        injector = AudioTagInjector()
        result = injector.inject(
            "Everything is terrible and confusing.",
            _agent_state(
                mood=-0.9,
                energy=0.1,
                arousal=0.9,
                frustration=0.9,
                identity_drift=0.5,
                prev_mood=0.2,
            ),
            _luna_profile(),
        )

        tag_count = result.count("[")
        assert tag_count <= 2

    def test_disallowed_tag_filtered_out(self) -> None:
        """Tags not in the voice's allowlist are suppressed."""
        injector = AudioTagInjector()
        profile = VoiceProfile(
            voice_id="test_id",
            agent_key="test",
            base_stability=0.5,
            base_speed=1.0,
            allowed_tags=frozenset(),
        )
        result = injector.inject(
            "I'm so sad.",
            _agent_state(mood=-0.8, energy=0.1),
            profile,
        )

        assert "[" not in result

    def test_strip_removes_all_tags(self) -> None:
        """Strip function produces clean display text."""
        assert (
            AudioTagInjector.strip("[sighs] I feel so tired. [pauses]")
            == "I feel so tired."
        )

    def test_strip_handles_no_tags(self) -> None:
        """Strip on clean text is a no-op."""
        assert AudioTagInjector.strip("Hello world.") == "Hello world."


# ── AudioProvider / ElevenLabsProvider ────────────────────────────


class TestAudioProvider:
    """Verify the provider adapter interface and error handling."""

    def test_result_dataclass(self) -> None:
        """AudioResult holds chunks and request id."""
        result = AudioResult(chunks=[b"chunk1", b"chunk2"], request_id="req-123")
        assert len(result.chunks) == 2
        assert result.request_id == "req-123"

    def test_model_selection_act1_uses_v3(self) -> None:
        """Acts 1–2 select eleven_v3 model."""
        provider = ElevenLabsProvider(api_key="test")
        assert provider.model_for_act(1) == "eleven_v3"
        assert provider.model_for_act(2) == "eleven_v3"

    def test_model_selection_act3_uses_flash(self) -> None:
        """Act 3 selects eleven_flash_v2_5 model."""
        provider = ElevenLabsProvider(api_key="test")
        assert provider.model_for_act(3) == "eleven_flash_v2_5"

    def test_output_format_act1_high_quality(self) -> None:
        """Acts 1–2 use high-quality mp3."""
        provider = ElevenLabsProvider(api_key="test")
        assert provider.output_format_for_act(1) == "mp3_44100_128"

    def test_output_format_act3_low_latency(self) -> None:
        """Act 3 uses low-latency mp3."""
        provider = ElevenLabsProvider(api_key="test")
        assert provider.output_format_for_act(3) == "mp3_22050_32"

    def test_synthesize_delegates_to_client(self) -> None:
        """Synthesize calls the underlying client and returns chunks."""
        mock_client = MagicMock()
        mock_client.text_to_speech.convert_as_stream.return_value = iter([b"a", b"b"])
        provider = ElevenLabsProvider(api_key="test", client=mock_client)
        settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.1,
            speed=1.0,
            use_speaker_boost=True,
        )

        result = provider.synthesize(
            voice_id="luna_id",
            text="Hello.",
            model_id="eleven_v3",
            settings=settings,
            output_format="mp3_44100_128",
        )

        assert result.chunks == [b"a", b"b"]
        mock_client.text_to_speech.convert_as_stream.assert_called_once()

    def test_synthesize_timeout_returns_empty(self) -> None:
        """Provider returns empty result on client error."""
        mock_client = MagicMock()
        mock_client.text_to_speech.convert_as_stream.side_effect = TimeoutError("boom")
        provider = ElevenLabsProvider(api_key="test", client=mock_client)
        settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.1,
            speed=1.0,
            use_speaker_boost=True,
        )

        result = provider.synthesize(
            voice_id="luna_id",
            text="Hello.",
            model_id="eleven_v3",
            settings=settings,
            output_format="mp3_44100_128",
        )

        assert result.chunks == []
        assert result.request_id is None

    def test_previous_request_ids_passed(self) -> None:
        """Request stitching IDs are forwarded to the client."""
        mock_client = MagicMock()
        mock_client.text_to_speech.convert_as_stream.return_value = iter([b"x"])
        provider = ElevenLabsProvider(api_key="test", client=mock_client)
        settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.1,
            speed=1.0,
            use_speaker_boost=True,
        )

        provider.synthesize(
            voice_id="luna_id",
            text="Continuation.",
            model_id="eleven_v3",
            settings=settings,
            output_format="mp3_44100_128",
            previous_request_ids=["req-001", "req-002"],
        )

        call_kwargs = mock_client.text_to_speech.convert_as_stream.call_args.kwargs
        assert call_kwargs["previous_request_ids"] == [
            "req-001",
            "req-002",
        ]

    def test_loads_api_key_from_env(self) -> None:
        """Provider reads ELEVENLABS_API_KEY from environment."""
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "sk-test"}):
            provider = ElevenLabsProvider()
            assert provider._api_key == "sk-test"

    def test_missing_api_key_raises(self) -> None:
        """Provider raises when env key is absent."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ElevenLabsConfigError, match="ELEVENLABS_API_KEY"),
        ):
            ElevenLabsProvider()

    def test_explicit_key_overrides_env(self) -> None:
        """Explicit api_key takes precedence over environment."""
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "env-key"}):
            provider = ElevenLabsProvider(api_key="explicit-key")
            assert provider._api_key == "explicit-key"
