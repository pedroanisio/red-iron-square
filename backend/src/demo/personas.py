"""Static personas, scenarios, and display mappings for the demo.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from src.demo.models import DemoPersona, DemoScenario

DEFAULT_PERSONAS: dict[str, DemoPersona] = {
    "luna": DemoPersona(
        key="luna",
        name="Luna",
        summary="Thoughtful. Cautious. Feels things deeply.",
        traits={
            "O": 0.3,
            "C": 0.7,
            "E": 0.3,
            "A": 0.8,
            "N": 0.8,
            "R": 0.3,
            "I": 0.7,
            "T": 0.7,
        },
    ),
    "marco": DemoPersona(
        key="marco",
        name="Marco",
        summary="Curious. Bold. Bounces back fast.",
        traits={
            "O": 0.9,
            "C": 0.3,
            "E": 0.8,
            "A": 0.5,
            "N": 0.2,
            "R": 0.9,
            "I": 0.4,
            "T": 0.2,
        },
    ),
}

SCRIPTED_SCENARIOS: dict[str, DemoScenario] = {
    "promotion": DemoScenario(
        key="promotion",
        name="The Promotion",
        description=(
            "Offered a major career promotion that requires relocating "
            "to an unfamiliar city."
        ),
        values={
            "O": 0.8,
            "C": 0.5,
            "E": 0.6,
            "A": 0.4,
            "N": 0.6,
            "R": 0.5,
            "I": 0.7,
            "T": 0.6,
        },
        forced_outcome=0.3,
    ),
    "phone_call": DemoScenario(
        key="phone_call",
        name="The Phone Call",
        description=(
            "Best friend calls, upset, saying your decision feels like abandonment."
        ),
        values={
            "O": 0.1,
            "C": 0.3,
            "E": 0.8,
            "A": 0.9,
            "N": 0.8,
            "R": 0.4,
            "I": 0.6,
            "T": 0.5,
        },
        forced_outcome=-0.4,
    ),
    "three_months": DemoScenario(
        key="three_months",
        name="Three Months Later",
        description=(
            "Three months in the new city. Still no close friends. "
            "Saturday nights are silent."
        ),
        values={
            "O": 0.5,
            "C": 0.3,
            "E": 0.7,
            "A": 0.6,
            "N": 0.7,
            "R": 0.6,
            "I": 0.5,
            "T": 0.4,
        },
        forced_outcome=-0.2,
    ),
}

DISPLAY_EMOTION_LABELS = {
    "EXCITEMENT": "Excitement",
    "ENTHUSIASM": "Enthusiasm",
    "CONTENTMENT": "Contentment",
    "FRUSTRATION_EMO": "Frustration",
    "ANXIETY": "Anxiety",
    "BOREDOM": "Boredom",
    "FLOW": "Focus",
    "RESIGNATION": "Resignation",
    "RELIEF": "Relief",
    "PRIDE": "Pride",
    "SHAME": "Shame",
    "AUTHENTICITY": "Feeling like myself",
    "IDENTITY_THREAT": "Inner conflict",
    "IDENTITY_CRISIS": "Lost",
}
