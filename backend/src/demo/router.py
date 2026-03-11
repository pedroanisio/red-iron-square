"""FastAPI routes for the Two Minds demo.

DISCLAIMER: No information within should be taken for granted.
Any statement or premise not backed by a real logical definition
or verifiable reference may be invalid, erroneous, or a hallucination.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src.demo.schemas import DemoCustomScenarioRequest, DemoSessionCreateRequest
from src.demo.service import DemoSessionService


def create_demo_router(service: DemoSessionService) -> APIRouter:
    """Build demo routes."""
    router = APIRouter(tags=["demo"])

    @router.post("/demo/sessions")
    def create_session(body: DemoSessionCreateRequest) -> dict[str, Any]:
        return {"data": service.create_session(body.act_number)}

    @router.get("/demo/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        try:
            return {"data": service.get_session(session_id)}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Demo session not found.",
            ) from exc

    @router.post("/demo/sessions/{session_id}/scripted/{scenario_key}")
    def run_scripted(session_id: str, scenario_key: str) -> dict[str, Any]:
        try:
            return {"data": service.run_scripted(session_id, scenario_key)}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Demo session not found.",
            ) from exc

    @router.post("/demo/sessions/{session_id}/scenarios")
    def run_custom(
        session_id: str,
        body: DemoCustomScenarioRequest,
    ) -> dict[str, Any]:
        try:
            return {"data": service.run_custom(session_id, body.text)}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Demo session not found.",
            ) from exc

    @router.post("/demo/sessions/{session_id}/swap")
    def swap_personalities(session_id: str) -> dict[str, Any]:
        try:
            return {"data": service.swap_personalities(session_id)}
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Demo session not found.",
            ) from exc

    @router.websocket("/demo/sessions/{session_id}/stream")
    async def stream_session(websocket: WebSocket, session_id: str) -> None:
        try:
            queue = service.subscribe(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(
                    {
                        "event_type": event.event_type,
                        "session_id": event.session_id,
                        "payload": event.payload,
                    }
                )
        except WebSocketDisconnect:
            return
        finally:
            service.unsubscribe(session_id, queue)

    return router
