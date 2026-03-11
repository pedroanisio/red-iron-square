"""Command-line interface for the public SDK surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from src.sdk import AgentSDK


def _load_json_argument(raw: str) -> Any:
    """Load JSON either inline or from an @-prefixed file path."""
    if raw.startswith("@"):
        return json.loads(Path(raw[1:]).read_text(encoding="utf-8"))
    return json.loads(raw)


def _build_actions(sdk: AgentSDK, payload: list[dict[str, Any]]) -> list[Any]:
    """Build SDK actions from JSON payloads."""
    return [
        sdk.action(
            item["name"],
            item["modifiers"],
            description=item.get("description", ""),
        )
        for item in payload
    ]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common JSON input arguments shared by commands."""
    parser.add_argument("--personality", required=True, help="JSON object or @file path.")
    parser.add_argument("--scenario", required=True, help="JSON object or @file path.")
    parser.add_argument("--actions", required=True, help="JSON array or @file path.")
    parser.add_argument("--temperature", type=float, default=1.0)


def _handle_decide(args: argparse.Namespace) -> int:
    """Run a one-shot SDK decision and print JSON."""
    sdk = AgentSDK.default()
    personality = sdk.personality(_load_json_argument(args.personality))
    scenario_payload = _load_json_argument(args.scenario)
    actions = _build_actions(sdk, _load_json_argument(args.actions))
    scenario = sdk.scenario(
        scenario_payload["values"],
        name=scenario_payload.get("name", ""),
        description=scenario_payload.get("description", ""),
    )
    result = sdk.decide(
        personality,
        scenario,
        actions,
        temperature=args.temperature,
    )
    print(result.model_dump_json(indent=2))
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    """Run a temporal or self-aware simulation and print JSON."""
    sdk = AgentSDK.default()
    personality = sdk.personality(_load_json_argument(args.personality))
    actions = _build_actions(sdk, _load_json_argument(args.actions))
    scenario_payloads = _load_json_argument(args.scenarios)
    scenarios = [
        sdk.scenario(
            item["values"],
            name=item.get("name", ""),
            description=item.get("description", ""),
        )
        for item in scenario_payloads
    ]
    outcomes = _load_json_argument(args.outcomes) if args.outcomes else None

    trace: BaseModel
    if args.self_model:
        initial_self_model = sdk.initial_self_model(_load_json_argument(args.self_model))
        sa_simulator = sdk.self_aware_simulator(
            personality,
            initial_self_model,
            actions,
            temperature=args.temperature,
        )
        trace = sa_simulator.run(scenarios, outcomes=outcomes)
    else:
        t_simulator = sdk.simulator(
            personality,
            actions,
            temperature=args.temperature,
        )
        trace = t_simulator.run(scenarios, outcomes=outcomes)
    print(trace.model_dump_json(indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="red-iron-square")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide_parser = subparsers.add_parser("decide", help="Run one SDK decision.")
    _add_common_arguments(decide_parser)
    decide_parser.set_defaults(handler=_handle_decide)

    simulate_parser = subparsers.add_parser("simulate", help="Run a simulation trace.")
    simulate_parser.add_argument("--personality", required=True, help="JSON object or @file path.")
    simulate_parser.add_argument("--actions", required=True, help="JSON array or @file path.")
    simulate_parser.add_argument("--scenarios", required=True, help="JSON array or @file path.")
    simulate_parser.add_argument("--outcomes", help="JSON array or @file path.")
    simulate_parser.add_argument(
        "--self-model",
        help="JSON object or @file path for self-aware simulation.",
    )
    simulate_parser.add_argument("--temperature", type=float, default=1.0)
    simulate_parser.set_defaults(handler=_handle_simulate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.handler(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
