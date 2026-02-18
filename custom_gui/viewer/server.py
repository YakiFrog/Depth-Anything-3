from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json
import asyncio
import os

app = FastAPI()

# Global state to store the latest depth and rgb data
latest_data = {
    "depth": None,
    "rgb": None,
    "width": 0,
    "height": 0
}

active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text() # Just keep connection alive
    except Exception as e:
        print(f"WS Error/Closed: {e}")
        active_connections.remove(websocket)

async def broadcast_data(data):
    for connection in active_connections:
        await connection.send_text(json.dumps(data))

@app.post("/update")
async def update_data(data: dict):
    global latest_data
    latest_data = data
    # print(f"Received update. Keys: {data.keys()}")
    await broadcast_data(data)
    return {"status": "ok"}

script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(script_dir, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
