"""ElevenLabs audio integration for the Two Minds demo.

Provides voice-settings calculation from agent state, audio-tag
injection/stripping, and a thin provider adapter around the
ElevenLabs SDK with timeout and error normalization.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from src.shared.logging import get_logger

log = get_logger(module="demo.audio")

# ── Value objects ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Per-agent voice identity and tag allowlist."""

    voice_id: str
    agent_key: str
    base_stability: float
    base_speed: float
    allowed_tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    """One-to-one mapping to the ElevenLabs voice_settings object."""

    stability: float
    similarity_boost: float
    style: float
    speed: float
    use_speaker_boost: bool = True


@dataclass(frozen=True, slots=True)
class AudioResult:
    """Collected audio chunks from one TTS call."""

    chunks: list[bytes] = field(default_factory=list)
    request_id: str | None = None


class ElevenLabsConfigError(Exception):
    """Raised when ElevenLabs API key is missing."""


# ── Voice settings calculator ─────────────────────────────────────

_SIMILARITY_BOOST = 0.75
_SPEED_RANGE = 0.08
_STYLE_CEIL = 0.50


class VoiceSettingsCalculator:
    """Map agent simulation state to ElevenLabs voice parameters."""

    def compute(
        self,
        state: dict[str, float],
        profile: VoiceProfile,
        *,
        act_number: int = 1,
    ) -> VoiceSettings:
        """Derive per-call voice settings from agent state."""
        mood = state.get("mood", 0.0)
        energy = state.get("energy", 0.5)
        arousal = state.get("arousal", 0.3)

        emotional_intensity = min(1.0, arousal + abs(mood))
        stability = _clamp(
            profile.base_stability - emotional_intensity * 0.25,
            lo=0.15,
            hi=0.85,
        )
        style = _clamp(abs(mood) * _STYLE_CEIL, lo=0.0, hi=_STYLE_CEIL)
        speed = _clamp(
            profile.base_speed + (energy - 0.5) * _SPEED_RANGE * 2,
            lo=0.5,
            hi=2.0,
        )
        use_speaker_boost = act_number < 3

        return VoiceSettings(
            stability=round(stability, 3),
            similarity_boost=_SIMILARITY_BOOST,
            style=round(style, 3),
            speed=round(speed, 3),
            use_speaker_boost=use_speaker_boost,
        )


# ── Audio tag injector ────────────────────────────────────────────

_TAG_RE = re.compile(r"\[.*?\]\s*")

_STATE_TAG_RULES: list[tuple[str, _TagCondition]] = []

# Type alias for condition functions checked against agent state.
_TagCondition = tuple[str, float, str]  # (field, threshold, comparison)


@dataclass(frozen=True, slots=True)
class _TagCandidate:
    """One potential audio tag with its triggering condition."""

    tag: str
    priority: int

    def matches(self, state: dict[str, float]) -> bool:
        """Check whether agent state triggers this tag."""
        raise NotImplementedError  # pragma: no cover


class _ThresholdCandidate(_TagCandidate):
    """Tag triggered when a field exceeds or drops below a threshold."""

    field: str
    threshold: float
    above: bool

    def __init__(
        self,
        tag: str,
        priority: int,
        *,
        field: str,
        threshold: float,
        above: bool = True,
    ) -> None:
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "above", above)

    def matches(self, state: dict[str, float]) -> bool:
        """Return True when agent state triggers this tag."""
        val = state.get(self.field, 0.0)
        return val > self.threshold if self.above else val < self.threshold


class _CompoundCandidate(_TagCandidate):
    """Tag triggered by a combination of state fields."""

    conditions: list[tuple[str, float, bool]]

    def __init__(
        self,
        tag: str,
        priority: int,
        *,
        conditions: list[tuple[str, float, bool]],
    ) -> None:
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "conditions", conditions)

    def matches(self, state: dict[str, float]) -> bool:
        """Return True when all compound conditions are met."""
        for field_name, threshold, above in self.conditions:
            val = state.get(field_name, 0.0)
            if above and val <= threshold:
                return False
            if not above and val >= threshold:
                return False
        return True


class _MoodDropCandidate(_TagCandidate):
    """Tag triggered by rapid mood decline between ticks."""

    drop_threshold: float

    def __init__(self, tag: str, priority: int, *, drop_threshold: float) -> None:
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "drop_threshold", drop_threshold)

    def matches(self, state: dict[str, float]) -> bool:
        """Return True when mood dropped sharply since previous tick."""
        prev = state.get("prev_mood")
        if prev is None:
            return False
        return (state.get("mood", 0.0) - prev) < self.drop_threshold


# Ordered by priority (lower = higher priority).
TAG_CANDIDATES: list[_TagCandidate] = [
    _MoodDropCandidate("voice breaking", priority=1, drop_threshold=-0.3),
    _CompoundCandidate(
        "sighs",
        priority=2,
        conditions=[("mood", -0.5, False), ("energy", 0.3, False)],
    ),
    _CompoundCandidate(
        "softly",
        priority=3,
        conditions=[("mood", -0.5, False), ("energy", 0.3, False)],
    ),
    _ThresholdCandidate("frustrated", priority=4, field="frustration", threshold=0.7),
    _ThresholdCandidate(
        "hesitantly", priority=5, field="identity_drift", threshold=0.3
    ),
    _CompoundCandidate(
        "excited",
        priority=6,
        conditions=[("mood", 0.5, True), ("arousal", 0.6, True)],
    ),
    _CompoundCandidate(
        "with confidence",
        priority=7,
        conditions=[("energy", 0.8, True), ("mood", 0.3, True)],
    ),
]

_MAX_TAGS = 2


class AudioTagInjector:
    """Inject and strip ElevenLabs v3 audio tags based on state."""

    def inject(
        self,
        text: str,
        state: dict[str, float],
        profile: VoiceProfile,
    ) -> str:
        """Insert up to two audio tags into narrative text."""
        selected: list[str] = []
        for candidate in sorted(TAG_CANDIDATES, key=lambda c: c.priority):
            if len(selected) >= _MAX_TAGS:
                break
            if candidate.tag not in profile.allowed_tags:
                continue
            if candidate.matches(state):
                selected.append(candidate.tag)

        if not selected:
            return text

        prefix = " ".join(f"[{tag}]" for tag in selected)
        return f"{prefix} {text}"

    @staticmethod
    def strip(text: str) -> str:
        """Remove all square-bracket audio tags from text."""
        return _TAG_RE.sub("", text).strip()


# ── ElevenLabs provider adapter ───────────────────────────────────

_MODEL_V3 = "eleven_v3"
_MODEL_FLASH = "eleven_flash_v2_5"
_FORMAT_HQ = "mp3_44100_128"
_FORMAT_FAST = "mp3_22050_32"


class ElevenLabsProvider:
    """Thin adapter over the ElevenLabs Python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key or self._load_api_key()
        self._client = client

    @staticmethod
    def _load_api_key() -> str:
        """Resolve API key from environment, raising on absence."""
        load_dotenv()
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise ElevenLabsConfigError(
                "ElevenLabs credentials not configured. "
                "Set `ELEVENLABS_API_KEY` in .env or environment."
            )
        return key

    def model_for_act(self, act_number: int) -> str:
        """Select TTS model based on act requirements."""
        return _MODEL_FLASH if act_number >= 3 else _MODEL_V3

    def output_format_for_act(self, act_number: int) -> str:
        """Select output format based on latency needs."""
        return _FORMAT_FAST if act_number >= 3 else _FORMAT_HQ

    def synthesize(
        self,
        *,
        voice_id: str,
        text: str,
        model_id: str,
        settings: VoiceSettings,
        output_format: str,
        previous_request_ids: list[str] | None = None,
    ) -> AudioResult:
        """Call ElevenLabs HTTP streaming endpoint."""
        client = self._resolve_client()
        try:
            stream = client.text_to_speech.convert_as_stream(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
                voice_settings={
                    "stability": settings.stability,
                    "similarity_boost": settings.similarity_boost,
                    "style": settings.style,
                    "speed": settings.speed,
                    "use_speaker_boost": settings.use_speaker_boost,
                },
                previous_request_ids=previous_request_ids or [],
            )
            chunks = list(stream)
            return AudioResult(chunks=chunks, request_id=None)
        except Exception:
            log.warning("elevenlabs_synthesis_failed", voice_id=voice_id)
            return AudioResult()

    def _resolve_client(self) -> Any:
        """Lazy-init the ElevenLabs SDK client."""
        if self._client is None:
            from elevenlabs import ElevenLabs  # type: ignore[import-not-found]

            self._client = ElevenLabs(api_key=self._api_key)
        return self._client


# ── Helpers ───────────────────────────────────────────────────────


def _clamp(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
