import os
import json
import asyncio
import threading
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your agent components
from agent import app as langgraph_app
from agent import NetworkAgentState, ml_intel
from logger import run_simulator

app = FastAPI(title="Neurotech NOC Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
LOG_FILE = "network_telemetry.log"
# Global buffer for broadcasting to multiple clients
telemetry_buffer = [] 
agent_log_buffer = [] 
approval_request = asyncio.Event()
pending_approval_data = {}

# --- 1. Telemetry Streamer (Reads log file) ---
async def tail_log_file():
    """Tails the telemetry log file and pushes new lines to the buffer."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f: f.write("")
        
    with open(LOG_FILE, "r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.5)
                continue
            if "{" in line:
                try:
                    data = json.loads(line[line.find("{"):])
                    telemetry_buffer.append(data)
                    # Keep buffer size manageable
                    if len(telemetry_buffer) > 500: telemetry_buffer.pop(0)
                except: pass

# --- 2. Background Agent Runner ---
async def run_autonomous_agent():
    """Runs the agent loop and intercepts the Sentry node for Web approvals."""
    cycle = 0
    while True:
        cycle += 1
        print(f"\n--- 🤖 STARTING AGENT CYCLE #{cycle} ---")
        thread_id = f"web-ops-{cycle}"
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "latest_telemetry": [], "ml_results": [], "network_health_metrics": {},
            "current_hypothesis": "", "is_anomaly_detected": False, "failure_type": "normal",
            "risk_score": 0.0, "blast_radius": "LOCAL", "affected_devices": [],
            "next_action": None, "decision_args": None, "auto_approved": True,
            "reasoning_log": [], "action_history": [], "ml_insights": []
        }

        try:
            print("DEBUG: Calling langgraph_app.astream...")
            async for event in langgraph_app.astream(initial_state, config=config, stream_mode="values"):
                if event.get("reasoning_log"):
                    new_log = event["reasoning_log"][-1]
                    # FIX: Removed 'await' (list.append is synchronous)
                    agent_log_buffer.append({"type": "agent_step", "data": new_log})
                    print(f"DEBUG: Log generated: {new_log}")

                # Check for Sentry Interrupt
                snapshot = langgraph_app.get_state(config)
                if snapshot.next and snapshot.next[0] == "sentry":
                    global pending_approval_data
                    pending_approval_data = {
                        "thread_id": thread_id,
                        "action": event["next_action"],
                        "risk": event["risk_score"],
                        "reason": event["reasoning_log"][-1]
                    }
                    # FIX: Removed 'await'
                    agent_log_buffer.append({"type": "approval_required", "data": pending_approval_data})
                    
                    print("⚠️ AWAITING WEB APPROVAL...")
                    await approval_request.wait()
                    approval_request.clear()
                    await langgraph_app.ainvoke(None, config=config)

            print(f"--- 🤖 CYCLE #{cycle} COMPLETE ---")

        except Exception as e:
            # FIX: Removed 'await'
            agent_log_buffer.append({"type": "error", "data": str(e)})
            print(f"❌ AGENT ERROR: {str(e)}")
        
        if len(agent_log_buffer) > 200: agent_log_buffer.pop(0)
        await asyncio.sleep(8)

# --- 3. WebSocket Endpoints (Broadcast Pattern) ---

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    last_sent_index = len(telemetry_buffer)
    try:
        while True:
            while last_sent_index < len(telemetry_buffer):
                await websocket.send_json(telemetry_buffer[last_sent_index])
                last_sent_index += 1
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()
    last_sent_index = len(agent_log_buffer)
    try:
        while True:
            while last_sent_index < len(agent_log_buffer):
                await websocket.send_json(agent_log_buffer[last_sent_index])
                last_sent_index += 1
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

# --- 4. Human-in-the-loop API ---

@app.post("/api/approve")
async def approve_action(decision: Dict):
    if not decision.get("approved"):
        config = {"configurable": {"thread_id": decision.get("thread_id")}}
        langgraph_app.update_state(config, {"next_action": "MONITOR", "auto_approved": True})
    
    approval_request.set()
    return {"status": "resumed"}

@app.get("/")
async def root():
    """Simple health-check endpoint so Render's health checks (and you) can confirm the service is up."""
    return {
        "status": "ok",
        "service": "Neurotech NOC Backend",
        "telemetry_records_buffered": len(telemetry_buffer),
        "agent_log_entries": len(agent_log_buffer),
    }


@app.on_event("startup")
async def startup_event():
    # Telemetry generation is blocking (time.sleep-based), so it runs in its
    # own daemon thread rather than as an asyncio task — this is what makes
    # the whole system self-sufficient on Render: no separate process or
    # manual `python logger.py` step is needed.
    threading.Thread(target=run_simulator, kwargs={"verbose": True}, daemon=True).start()
    asyncio.create_task(tail_log_file())
    asyncio.create_task(run_autonomous_agent())


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")