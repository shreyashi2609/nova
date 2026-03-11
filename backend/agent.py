import os
import json
import operator
import time
import threading
import traceback
from typing import Annotated, List, Optional, TypedDict
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from ml_models import NetworkMLIntelligence

load_dotenv()

_api_key = os.getenv("GROQ_API_KEY", "").strip()
if _api_key:
    os.environ["GROQ_API_KEY"] = _api_key

llm = ChatOpenAI(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=os.getenv("GROQ_API_KEY") or "dummy",
    base_url="https://api.groq.com/openai/v1",
    temperature=0.15,
)

# Global ML intelligence
ml_intel = NetworkMLIntelligence()

def _sanitize_numpy(obj):
    """Recursively converts NumPy data types to native Python types for LangGraph serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_numpy(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)       
    else:
        return obj


# State Schema

class NetworkAgentState(TypedDict):
    # Raw data
    latest_telemetry:      List[dict]
    ml_results:            List[dict]          # per-record ML inference output
    network_health_metrics: dict

    # Reasoning outputs
    current_hypothesis:    str
    is_anomaly_detected:   bool
    failure_type:          str                 # ML predicted class
    risk_score:            float               # 0.0 – 1.0
    blast_radius:          str                 # LOCAL / REGIONAL / CORE_WIDE
    affected_devices:      List[str]

    # Decision outputs
    next_action:           Optional[str]
    decision_args:         Optional[str]
    auto_approved:         bool

    # Audit trail (append-only via operator.add)
    reasoning_log:         Annotated[List[str], operator.add]
    action_history:        Annotated[List[str], operator.add]
    ml_insights:           Annotated[List[str], operator.add]

# Helpers

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _safe_llm(messages) -> str:
    """Call LLM; return empty string on any failure (no API key etc.)."""
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as exc:
        return f"[LLM_UNAVAILABLE: {exc}]"


# Node 1 — Observer

def observer_node(state: NetworkAgentState) -> dict:
    log_file = "network_telemetry.log"
    raw = []

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()[-80:]
        for line in lines:
            j = line.find('{')
            if j != -1:
                try:
                    raw.append(json.loads(line[j:]))
                except Exception:
                    pass

    if not raw:
        return {
            "latest_telemetry":      [],
            "ml_results":            [],
            "network_health_metrics":{},
            "reasoning_log":         [f"[{_ts()}] Observer: No telemetry data found."],
        }

    # Run ML inference on every record
    ml_results = [ml_intel.ingest(r) for r in raw]

    # Aggregate health metrics
    n        = len(raw)
    critical = [r["device_id"] for r in raw if r.get("status") == "CRITICAL"]
    degraded = [r["device_id"] for r in raw if r.get("status") == "DEGRADED"]
    avg_lat  = sum(r["metrics"]["latency_ms"]      for r in raw) / n
    avg_loss = sum(r["metrics"]["packet_loss_pct"] for r in raw) / n
    avg_util = sum(r["metrics"]["utilization_pct"] for r in raw) / n
    avg_cpu  = sum(r["metrics"]["cpu_util_pct"]    for r in raw) / n

    # High-risk ML signals
    high_risk = [mr for mr in ml_results if mr.get("aggregate_risk_score", 0) >= 0.60]
    early_warnings = [
        mr["forecast"]["device_id"]
        for mr in ml_results
        if mr.get("forecast") and mr["forecast"].get("early_warning")
    ]

    # Best ML prediction (highest confidence, highest risk)
    best_pred  = None
    best_score = 0.0
    for mr in ml_results:
        p = mr.get("prediction")
        if p and p["risk_score"] > best_score:
            best_score = p["risk_score"]
            best_pred  = p

    failure_type = best_pred["predicted_class"] if best_pred else "normal"
    risk_score   = round(max((mr.get("aggregate_risk_score", 0) for mr in ml_results), default=0.0), 3)

    # Blast radius estimation
    unique_regions = list(set(r["region"] for r in raw if r.get("status") != "HEALTHY"))
    if len(unique_regions) >= 3:
        blast = "CORE_WIDE"
    elif len(unique_regions) == 2:
        blast = "REGIONAL"
    else:
        blast = "LOCAL"

    ml_log = []
    if high_risk:
        ml_log.append(f"[{_ts()}] ML: {len(high_risk)} high-risk signals detected.")
    if early_warnings:
        ml_log.append(f"[{_ts()}] ML Forecast: Rising latency trend on {early_warnings}.")
    if best_pred:
        ml_log.append(
            f"[{_ts()}] ML Classifier: '{failure_type}' "
            f"(confidence={best_pred['confidence']:.0%}, risk={best_score:.2f})"
        )

    # Wrap the entire return dictionary
    return _sanitize_numpy({
        "latest_telemetry":       raw,
        "ml_results":             ml_results,
        "failure_type":           failure_type,
        "risk_score":             risk_score,
        "blast_radius":           blast,
        "affected_devices":       list(set(critical + degraded))[:10],
        "network_health_metrics": {
            "total_signals":    n,
            "critical_count":   len(set(critical)),
            "degraded_count":   len(set(degraded)),
            "avg_latency_ms":   round(avg_lat, 1),
            "avg_loss_pct":     round(avg_loss, 3),
            "avg_util_pct":     round(avg_util, 1),
            "avg_cpu_pct":      round(avg_cpu, 1),
            "high_risk_signals":len(high_risk),
            "early_warnings":   early_warnings,
        },
        "reasoning_log": [
            f"[{_ts()}] Observer: Ingested {n} signals | "
            f"CRITICAL={len(set(critical))} DEGRADED={len(set(degraded))} "
            f"| Risk={risk_score:.2f} | Blast={blast}"
        ],
        "ml_insights": ml_log,
    })


# Node 2 — Reasoner

def reasoner_node(state: NetworkAgentState) -> dict:
    m       = state["network_health_metrics"]
    if not m:
        return {
            "current_hypothesis": "No telemetry.",
            "is_anomaly_detected": False,
            "reasoning_log": [f"[{_ts()}] Reasoner: Skipped — no data."],
        }

    sample  = json.dumps(state["latest_telemetry"][:6], indent=2)
    ml_ctx  = json.dumps({
        "ml_failure_type": state.get("failure_type", "unknown"),
        "risk_score":       state.get("risk_score", 0),
        "blast_radius":     state.get("blast_radius", "LOCAL"),
        "affected_devices": state.get("affected_devices", []),
    }, indent=2)

    prompt = f"""
You are the Senior Network Operations Engineer for Neurotech ISP.
Your job: Analyse incoming telemetry and form a precise technical hypothesis.

NETWORK SNAPSHOT (last 80 readings):
  Avg Latency       : {m['avg_latency_ms']} ms
  Avg Packet Loss   : {m['avg_loss_pct']} %
  Avg Utilization   : {m['avg_util_pct']} %
  Critical Devices  : {m['critical_count']}
  High-Risk Signals : {m['high_risk_signals']}

ML INTELLIGENCE:
{ml_ctx}

RAW TELEMETRY SAMPLE (first 6 records):
{sample}

TASK:
1. Identify the failure mode: Fiber Cut | BGP Flap | DDoS | Hardware Degradation | Peak Congestion | DNS Failure | Normal
2. Estimate blast radius: LOCAL (1 device) | REGIONAL (1 city) | CORE_WIDE (multi-region)
3. Describe SLA risk: how many customers are likely affected and for how long?
4. State your confidence (Low/Medium/High).

FORMAT YOUR RESPONSE EXACTLY AS:
Hypothesis: <one sentence root cause>
Failure Mode: <type>
Blast Radius: <LOCAL|REGIONAL|CORE_WIDE>
SLA Risk: <sentence>
Confidence: <Low|Medium|High>
Anomaly Detected: <Yes|No>
"""

    content = _safe_llm([
        SystemMessage(content="You are a senior ISP network engineer. Be concise and precise."),
        HumanMessage(content=prompt),
    ])

    def extract(tag: str, default: str = "") -> str:
        for line in content.split("\n"):
            if line.strip().startswith(tag + ":"):
                return line.split(":", 1)[1].strip()
        return default

    hypothesis   = extract("Hypothesis",   "No hypothesis formed.")
    is_anomaly   = "yes" in extract("Anomaly Detected", "No").lower()
    confidence   = extract("Confidence",   "Low")

    return {
        "current_hypothesis":  hypothesis,
        "is_anomaly_detected": is_anomaly,
        "reasoning_log": [
            f"[{_ts()}] Reasoner: {hypothesis} | Anomaly={is_anomaly} | Conf={confidence}"
        ],
    }


# Node 3 — Decider

# Action catalogue
ACTIONS = {
    "reroute_traffic":       {"risk": "MEDIUM", "reversible": True,  "auto_approve_threshold": 0.85},
    "apply_acl_filter":      {"risk": "LOW",    "reversible": True,  "auto_approve_threshold": 0.90},
    "bgp_reset":             {"risk": "HIGH",   "reversible": False, "auto_approve_threshold": 0.70},
    "rate_limit_edge":       {"risk": "LOW",    "reversible": True,  "auto_approve_threshold": 0.92},
    "capacity_reallocation": {"risk": "MEDIUM", "reversible": True,  "auto_approve_threshold": 0.80},
    "config_rollback":       {"risk": "HIGH",   "reversible": False, "auto_approve_threshold": 0.65},
    "dns_failover":          {"risk": "MEDIUM", "reversible": True,  "auto_approve_threshold": 0.85},
    "escalate_to_engineer":  {"risk": "NONE",   "reversible": True,  "auto_approve_threshold": 1.00},
    "MONITOR":               {"risk": "NONE",   "reversible": True,  "auto_approve_threshold": 1.00},
}

FAILURE_TO_ACTION = {
    "fiber_cut":             "reroute_traffic",
    "bgp_flap":              "bgp_reset",
    "ddos_attack":           "apply_acl_filter",
    "hardware_degradation":  "escalate_to_engineer",
    "peak_congestion":       "capacity_reallocation",
    "dns_failure":           "dns_failover",
    "link_flap":             "reroute_traffic",
    "normal":                "MONITOR",
}


def decider_node(state: NetworkAgentState) -> dict:
    # If no anomaly, OR if the anomaly is already mitigated (no affected devices left)
    if not state.get("is_anomaly_detected") or not state.get("affected_devices"):
        return {
            "next_action":    "MONITOR",
            "auto_approved":  True,
            "decision_args":  json.dumps({}),
            "reasoning_log":  [f"[{_ts()}] Decider: All within SLA or already mitigated — monitoring."],
        }
    
    failure_type = state.get("failure_type", "normal")
    risk_score   = state.get("risk_score", 0.5)
    blast        = state.get("blast_radius", "LOCAL")

    affected = state.get("affected_devices", [])
    primary_target = affected[0] if affected else "unknown_device" 

    # ML-first action selection
    ml_action = FAILURE_TO_ACTION.get(failure_type, "escalate_to_engineer")

    # LLM validates and may refine
    prompt = f"""
You are the Network Controller for Neurotech ISP.

TARGET DEVICE: {primary_target}

SITUATION:
  Hypothesis     : {state['current_hypothesis']}
  ML Failure Type: {failure_type}
  Risk Score     : {risk_score:.2f}
  Blast Radius   : {blast}
  Affected       : {state.get('affected_devices', [])}

ML SUGGESTED ACTION: {ml_action}

AVAILABLE ACTIONS (with risk level):
  reroute_traffic       — MEDIUM risk, reversible  (fiber cut, congestion)
  apply_acl_filter      — HIGH risk,    reversible  (DDoS mitigation)
  bgp_reset             — HIGH risk,   irreversible (BGP instability)
  rate_limit_edge       — LOW risk,    reversible  (DDoS pre-emptive)
  capacity_reallocation — MEDIUM risk, reversible  (congestion)
  config_rollback       — HIGH risk,   irreversible (misconfiguration)
  dns_failover          — MEDIUM risk, reversible  (DNS failure)
  escalate_to_engineer  — NONE risk                (hardware, thermal)
  MONITOR               — no action needed

RULES:
- Prefer LOW risk actions unless situation demands otherwise.
- For CORE_WIDE blast radius: always escalate in addition to the primary action.
- For risk_score >= 0.9: mandatory escalate_to_engineer notification.

FORMAT:
Primary Action: <action_name>
Reason: <one sentence>
Args: {{"device_id": "{primary_target}", "failure": "{state.get('failure_type')}"}}
"""

    content = _safe_llm(prompt)

    def extract(tag: str, default: str = "") -> str:
        for line in content.split("\n"):
            if line.strip().startswith(tag + ":"):
                return line.split(":", 1)[1].strip()
        return default

    action = extract("Primary Action", ml_action).strip()
    if action not in ACTIONS:
        action = ml_action

    reason    = extract("Reason", "Automated intervention based on ML prediction.")
    args_str  = extract("Args", "{}")
    try:
        args = json.loads(args_str)
    except Exception:
        args = {}

    args.update({
        "device_id": primary_target,
        "failure_type": failure_type,
        "risk_score":   risk_score,
        "blast_radius": blast,
        "hypothesis":   state["current_hypothesis"],
    })

    print("------------------------------------PRIMARY TARGET--------------------------------")
    print(f"----------------------------------IT IS: {primary_target}------------------------")

    # Auto-approve logic based on confidence threshold
    action_meta    = ACTIONS.get(action, ACTIONS["escalate_to_engineer"])
    auto_threshold = action_meta["auto_approve_threshold"]
    auto_approved  = risk_score >= auto_threshold or action_meta["risk"] == "NONE"

    return {
        "next_action":   action,
        "decision_args": json.dumps(args),
        "auto_approved": auto_approved,
        "reasoning_log": [
            f"[{_ts()}] Decider: Action={action} | AutoApproved={auto_approved} | Reason: {reason}"
        ],
    }


# Node 4 — Sentry (Human-in-the-Loop gate)

def sentry_node(state: NetworkAgentState) -> dict:
    """
    Checkpoint node. LangGraph will interrupt_before this node when
    the action requires human approval. The operator can:
      - Approve  → graph resumes to executor
      - Modify   → operator patches state, then resumes
      - Reject   → operator calls graph.update_state with next_action="MONITOR"
    """
    return {
        "reasoning_log": [
            f"[{_ts()}] Sentry: AWAITING HUMAN APPROVAL for '{state['next_action']}' "
            f"(risk={state['risk_score']:.2f}, blast={state['blast_radius']})"
        ]
    }


# Node 5 — Executor

from tools import (
    reroute_traffic_tool,
    apply_acl_filter_tool,
    bgp_reset_tool,
    rate_limit_edge_tool,
    capacity_reallocation_tool,
    config_rollback_tool,
    dns_failover_tool,
    escalate_to_engineer_tool,
)


def _invoke_tool(tool_fn, **kwargs) -> dict:
    """
    Calls a LangChain @tool safely.
    Returns {"status": "SUCCESS"|"FAILED", "message": <str>}.
    """
    try:
        message = tool_fn.invoke(kwargs)
        # All tools return a plain result string
        status = "ESCALATED" if "ESCALATION SENT" in str(message) else "SUCCESS"
        return {"status": status, "message": str(message)}
    except Exception as exc:
        return {"status": "FAILED", "message": str(exc)}


def _run_action(action: str, args: dict) -> dict:
    """
    Maps the agent action name → real tool call with correct parameter mapping.
    Extracts the right fields from args (populated by the decider) for each tool.
    """
    # Grab the full list for tools that need multiple devices (like capacity reallocation)
    devices = args.get("affected_devices", [])

    # 1. First, check if the LLM or decider explicitely passed a device_id
    device_id = args.get("device_id")

    # 2. If it didn't, try to pull it from the devices list
    if not device_id and devices:
        device_id = devices[0]

    # # 3. If that still fails, default to unknown
    # if not device_id:
    #     device_id = "unknown_device"
    
    region     = args.get("region", args.get("blast_radius", "Unknown"))
    hypothesis = args.get("hypothesis", "Autonomous intervention.")
    risk       = args.get("risk_score", 0.5)
    failure    = args.get("failure_type", "unknown")

    if action == "reroute_traffic":
        return _invoke_tool(
            reroute_traffic_tool,
            device_id = device_id,
            region    = region,
            reason    = f"[{failure}] {hypothesis}",
        )

    elif action == "apply_acl_filter":
        return _invoke_tool(
            apply_acl_filter_tool,
            device_id   = device_id,
            region      = region,
            threat_type = failure.upper(),
            action_type = "DROP" if risk >= 0.90 else "RATE_LIMIT",
        )

    elif action == "bgp_reset":
        return _invoke_tool(
            bgp_reset_tool,
            peer_id    = device_id,
            region     = region,
            reason     = f"[{failure}] {hypothesis}",
            reset_type = "hard" if risk >= 0.85 else "soft",
        )

    elif action == "rate_limit_edge":
        # Scale limit inversely with risk: higher risk → tighter cap
        limit = round(max(1.0, 10.0 * (1.0 - risk)), 1)
        return _invoke_tool(
            rate_limit_edge_tool,
            device_id  = device_id,
            region     = region,
            limit_gbps = limit,
            reason     = f"[{failure}] Pre-emptive cap. {hypothesis}",
        )

    elif action == "capacity_reallocation":
        # Use the remaining affected devices as relief links; fall back to a generic name
        relief = devices[1:] if len(devices) > 1 else [f"relief_link_{region.lower()}"]
        return _invoke_tool(
            capacity_reallocation_tool,
            congested_link = device_id,
            region         = region,
            target_links   = ", ".join(relief),
            reason         = f"[{failure}] {hypothesis}",
        )

    elif action == "config_rollback":
        return _invoke_tool(
            config_rollback_tool,
            device_id   = device_id,
            region      = region,
            snapshot_id = "latest",
            reason      = f"[{failure}] {hypothesis}",
        )

    elif action == "dns_failover":
        backup = f"dns_resolver_{region.lower().replace(' ','_')}_backup"
        return _invoke_tool(
            dns_failover_tool,
            resolver_id      = device_id,
            region           = region,
            backup_resolver  = backup,
            reason           = f"[{failure}] {hypothesis}",
        )

    else:  # escalate_to_engineer (and any unrecognised action)
        severity = "CRITICAL" if risk >= 0.85 else "HIGH" if risk >= 0.65 else "MEDIUM"
        return _invoke_tool(
            escalate_to_engineer_tool,
            device_id          = device_id,
            region             = region,
            severity           = severity,
            hypothesis         = hypothesis,
            recommended_action = f"Investigate {failure} on {device_id}. Risk={risk:.2f}.",
        )


def executor_node(state: NetworkAgentState) -> dict:
    action = state.get("next_action", "MONITOR")
    args = json.loads(state.get("decision_args") or "{}")
    
    devices = state.get("affected_devices", [])
    device_id = args.get("device_id") or (devices[0] if devices else "unknown_device")
    
    args["device_id"] = device_id

    if action == "MONITOR":
        return {
            "reasoning_log":  [f"[{_ts()}] Executor: No action needed. Monitoring."],
            "action_history": [f"MONITOR @ {_ts()}"],
        }

    try:
        result  = _run_action(action, args)
        outcome = result["status"]
        message = result["message"]
    except Exception as exc:
        outcome = "FAILED"
        message = str(exc)

    # Grab most recent telemetry record for ML feedback
    recent_logs     = state.get("latest_telemetry", [])
    feedback_record = recent_logs[-1] if recent_logs else {"metrics": {}}

    # Feed outcome back into ML learning loop
    devices   = args.get("affected_devices", [])
    device_id = devices[0] if devices else "unknown"
    ml_intel.record_intervention(
        action     = action,
        device_id  = device_id,
        outcome    = outcome,
        record     = feedback_record,
        true_label = args.get("failure_type"),
    )

    return {
        "reasoning_log": [
            f"[{_ts()}] Executor: {action} → {outcome}: {message}"
        ],
        "action_history": [
            f"{action} | {outcome} | {args.get('failure_type','?')} | @ {_ts()}"
        ],
        "ml_insights": [
            f"[{_ts()}] Learn: Intervention recorded. "
            f"Stats={ml_intel.get_intervention_stats()}"
        ],
    }


# Node 6 — Learner

def learner_node(state: NetworkAgentState) -> dict:
    """
    Post-execution learning: analyses action_history to update confidence
    thresholds and surfaces recommendations for policy tuning.
    """
    stats = ml_intel.get_intervention_stats()
    recs  = []

    if stats["total_interventions"] > 5 and stats["success_rate"] < 0.70:
        recs.append("⚠ Success rate below 70%. Consider adjusting auto-approval thresholds.")
    if stats["success_rate"] >= 0.90:
        recs.append("✅ High success rate. Confidence in autonomous actions is strong.")

    return {
        "ml_insights": [
            f"[{_ts()}] Learner: total={stats['total_interventions']} "
            f"success_rate={stats['success_rate']:.0%} | {'; '.join(recs) or 'No recommendations.'}"
        ]
    }


# Routing Logic

def route_after_decider(state: NetworkAgentState) -> str:
    action       = state.get("next_action", "MONITOR")
    auto_approved = state.get("auto_approved", False)

    if action == "MONITOR":
        return END
    if auto_approved:
        return "executor"
    return "sentry"


# Graph Assembly

workflow = StateGraph(NetworkAgentState)

workflow.add_node("observer", observer_node)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("decider",  decider_node)
workflow.add_node("sentry",   sentry_node)
workflow.add_node("executor", executor_node)
workflow.add_node("learner",  learner_node)

workflow.set_entry_point("observer")
workflow.add_edge("observer", "reasoner")
workflow.add_edge("reasoner", "decider")
workflow.add_conditional_edges("decider", route_after_decider, {
    "sentry":    "sentry",
    "executor":  "executor",
    END:         END,
})
workflow.add_edge("sentry",   "executor")
workflow.add_edge("executor", "learner")
workflow.add_edge("learner",  END)

checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["sentry"],  # pause for human approval on HIGH-risk actions
)


# Continuous Run Loop

def run_agent_cycle(thread_id: str = "neurotech-ops-1") -> dict:
    """Execute one full agent cycle. Returns the final state."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: NetworkAgentState = {
        "latest_telemetry":      [],
        "ml_results":            [],
        "network_health_metrics":{},
        "current_hypothesis":    "",
        "is_anomaly_detected":   False,
        "failure_type":          "normal",
        "risk_score":            0.0,
        "blast_radius":          "LOCAL",
        "affected_devices":      [],
        "next_action":           None,
        "decision_args":         None,
        "auto_approved":         True,
        "reasoning_log":         [],
        "action_history":        [],
        "ml_insights":           [],
    }

    final_state = None
    for event in app.stream(initial_state, config=config, stream_mode="values"):
        final_state = event

    return final_state or initial_state


def start_continuous_loop(interval_seconds: int = 10):
    """Run the agent in a continuous background loop with interactive Human-in-the-Loop."""
    print(f"\n🤖 Neurotech Autonomous Agent STARTED (interval={interval_seconds}s)\n")
    cycle = 0
    
    while True:
        cycle += 1
        thread_id = f"neurotech-ops-{cycle}"
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n{'─'*60}")
        print(f"  CYCLE #{cycle} @ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
        print(f"{'─'*60}")
        
        try:
            # 1. Run the agent cycle
            state = run_agent_cycle(thread_id=thread_id)
            
            # Print the logs for this cycle
            for line in state.get("reasoning_log", []):
                print(f"  {line}")  
            for line in state.get("ml_insights", []):
                print(f"  {line}")
                
            # 2. Check if LangGraph paused execution for Human Approval
            current_snapshot = app.get_state(config)
            
            if current_snapshot.next and current_snapshot.next[0] == "sentry":
                print("\n" + "⚠️"*30)
                print("⏸️  AGENT PAUSED: HUMAN APPROVAL REQUIRED")
                print(f"Proposed Action: {state['next_action']} | Risk Score: {state['risk_score']}")
                
                # Wait for you to type in the terminal
                user_input = input("\nApprove this action? (y/n): ").strip().lower()
                
                if user_input == 'y':
                    print("✅ Approved by Operator. Resuming execution...")
                    # Passing None tells LangGraph to resume from where it paused
                    for _ in app.stream(None, config=config, stream_mode="values"):
                        pass 
                else:
                    print("❌ Rejected by Operator. Forcing 'MONITOR' state.")
                    # Manually overwrite the agent's state to prevent the action
                    app.update_state(config, {"next_action": "MONITOR"})
                    for _ in app.stream(None, config=config, stream_mode="values"):
                        pass
            
            # Print the final execution result
            final_snapshot = app.get_state(config)
            final_state = final_snapshot.values
            if final_state.get("action_history"):
                print(f"  📋 Final Outcome: {final_state['action_history'][-1]}")

        except KeyboardInterrupt:
            print("\n⏹  Agent stopped by operator.")
            break
        except Exception:
            print(f"  ❌ Agent cycle error:\n{traceback.format_exc()}")
            
        time.sleep(interval_seconds)
        
if __name__ == "__main__":
    start_continuous_loop(interval_seconds=8)
