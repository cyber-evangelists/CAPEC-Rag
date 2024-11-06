from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from typing import Dict, Set, Any
import asyncio
from loguru import logger


class ConnectionManager:
    def __init__(self, max_connections: int = 100):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self.max_connections = max_connections
        self._lock = asyncio.Lock()  # For thread-safe operations

    async def connect(self, websocket: WebSocket) -> bool:
        """
        Attempt to establish a new connection if under max limit
        Returns True if connection was accepted, False otherwise
        """
        async with self._lock:
            if len(self.active_connections) >= self.max_connections:
                logger.warning(f"Connection rejected: Maximum connections ({self.max_connections}) reached")
                await websocket.close(
                    code=status.WS_1008_POLICY_VIOLATION,
                    reason=f"Server at maximum capacity ({self.max_connections} connections)",
                )
                return False

            await websocket.accept()
            self.active_connections[websocket] = {
                "connected_at": asyncio.get_event_loop().time(),
                "last_activity": asyncio.get_event_loop().time(),
            }
            logger.info(f"New connection accepted. Active connections: {len(self.active_connections)}")
            return True

    async def update_activity(self, websocket: WebSocket):
        """Update last activity timestamp for a connection"""
        if websocket in self.active_connections:
            self.active_connections[websocket][
                "last_activity"
            ] = asyncio.get_event_loop().time()

    def get_connection_count(self) -> int:
        """Get current number of active connections"""
        return len(self.active_connections)

    async def cleanup_inactive_connections(self, inactive_timeout: int = 300):
        """Cleanup connections that haven't had activity for the specified timeout"""
        current_time = asyncio.get_event_loop().time()
        async with self._lock:
            inactive_websockets = [
                ws
                for ws, data in self.active_connections.items()
                if current_time - data["last_activity"] > inactive_timeout
            ]

            for ws in inactive_websockets:
                try:
                    await ws.close(
                        code=status.WS_1000_NORMAL_CLOSURE,
                        reason="Connection closed due to inactivity",
                    )
                    self.active_connections.pop(ws)
                except Exception as e:
                    logger.error(f"Error closing inactive connection: {e}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                if self.active_connections[websocket]["heartbeat_task"]:
                    self.active_connections[websocket]["heartbeat_task"].cancel()
                self.active_connections.pop(websocket)

    async def _heartbeat(self, websocket: WebSocket):
        """Send periodic heartbeat to keep connection alive"""
        try:
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                try:
                    await websocket.send_json({"type": "ping"})
                    self.active_connections[websocket]["last_ping"] = asyncio.get_event_loop().time()
                except Exception:
                    break
        except asyncio.CancelledError:
            pass
