"""
Neurotech ISP Telemetry Simulator — Enhanced
Generates rich, realistic network telemetry for the autonomous agent.
Includes cascading failures, correlated device events, and anomaly bursts.
"""

import time
import random
import json
import logging
import os
import math
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# --- CONFIGURATION ---
LOG_FILE = "network_telemetry.log"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 3

# Setup Rotating Logger
logger = logging.getLogger("NetworkSimulator")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- ENHANCED NETWORK TOPOLOGY ---
# Simulating Neurotech's full multi-city ISP infrastructure
DEVICES = [
    # Edge Routers (customer-facing)
    {"id": "edge_router_mumbai_01", "type": "router", "region": "Mumbai",    "base_latency": 12, "tier": "edge"},
    {"id": "edge_router_mumbai_02", "type": "router", "region": "Mumbai",    "base_latency": 14, "tier": "edge"},
    {"id": "edge_router_delhi_01",  "type": "router", "region": "Delhi",     "base_latency": 18, "tier": "edge"},
    {"id": "edge_router_pune_01",   "type": "router", "region": "Pune",      "base_latency": 15, "tier": "edge"},
    # Core Switches (backbone)
    {"id": "core_switch_delhi_05",  "type": "switch", "region": "Delhi",     "base_latency": 25, "tier": "core"},
    {"id": "core_switch_mumbai_03", "type": "switch", "region": "Mumbai",    "base_latency": 20, "tier": "core"},
    {"id": "core_switch_hyd_01",    "type": "switch", "region": "Hyderabad", "base_latency": 22, "tier": "core"},
    # Fiber Links (physical layer)
    {"id": "fiber_link_blr_chn",    "type": "link",   "region": "South_Ring","base_latency": 8,  "tier": "transport"},
    {"id": "fiber_link_mum_del",    "type": "link",   "region": "West_North","base_latency": 35, "tier": "transport"},
    {"id": "fiber_link_del_hyd",    "type": "link",   "region": "Central",   "base_latency": 28, "tier": "transport"},
    # BGP Peering
    {"id": "bgp_peer_as1234",       "type": "peering","region": "Global",    "base_latency": 45, "tier": "peering"},
    {"id": "bgp_peer_as5678",       "type": "peering","region": "Global",    "base_latency": 38, "tier": "peering"},
    # DNS / CDN Nodes
    {"id": "dns_resolver_mumbai",   "type": "dns",    "region": "Mumbai",    "base_latency": 5,  "tier": "service"},
    {"id": "cdn_node_delhi_02",     "type": "cdn",    "region": "Delhi",     "base_latency": 10, "tier": "service"},
]

def _is_mitigated(device_id: str, scenario: str) -> bool:
    """Checks the agent's JSON state files to see if a fix was applied."""
    try:
        # Check Routing fixes (Fiber cuts, Congestion)
        if scenario in ("fiber_cut", "link_flap", "peak_congestion"):
            if os.path.exists("network_routing.json"):
                with open("network_routing.json", "r") as f:
                    routes = json.load(f)
                    if device_id in routes and routes[device_id].get("status") == "REROUTED":
                        return True
                        
        # Check ACL fixes (DDoS)
        elif scenario == "ddos_attack":
            if os.path.exists("acl_policy.json"):
                with open("acl_policy.json", "r") as f:
                    acls = json.load(f)
                    for rule in acls:
                        if rule.get("device_id") == device_id and rule.get("active"):
                            return True
                            
        # Check BGP fixes
        elif scenario == "bgp_flap":
            if os.path.exists("bgp_state.json"):
                with open("bgp_state.json", "r") as f:
                    bgp = json.load(f)
                    if device_id in bgp and bgp[device_id].get("status") == "RESETTING":
                        return True
                        
        # Check DNS fixes
        elif scenario == "dns_failure":
            if os.path.exists("dns_config.json"):
                with open("dns_config.json", "r") as f:
                    dns = json.load(f)
                    # DNS fixes are keyed by region in your tools.py
                    for region, data in dns.items():
                        if data.get("primary_resolver") == device_id:
                            return True

    except Exception:
        pass # Ignore file read errors (e.g., if agent is writing to it at this exact microsecond)
        
    return False

def _apply_time_of_day_factor():
    """Simulate peak-hour load patterns (9am-11am, 6pm-10pm IST)."""
    hour = datetime.now(timezone.utc).hour + 5  # rough IST offset
    hour = hour % 24
    if 4 <= hour <= 6:    # 9-11am IST
        return 1.35
    elif 13 <= hour <= 17: # 6-10pm IST
        return 1.55
    elif 0 <= hour <= 2:   # late night
        return 0.60
    return 1.0

def generate_telemetry(scenario="normal", device_override=None):
    """
    Generates a single, richly detailed telemetry reading.
    Supports cascading failure propagation and time-of-day load patterns.
    """
    device = device_override or random.choice(DEVICES)
    tod_factor = _apply_time_of_day_factor()

    # --- Baseline (healthy) metrics ---
    status = "HEALTHY"
    latency       = device["base_latency"] + random.randint(-2, 5)
    packet_loss   = round(random.uniform(0.0, 0.05), 3)
    utilization   = round(random.uniform(30.0, 60.0) * tod_factor, 1)
    cpu_temp      = random.randint(38, 55)
    cpu_util      = round(random.uniform(15.0, 40.0), 1)
    throughput_gb = round(random.uniform(1.0, 8.0) * tod_factor, 2)
    error_rate    = round(random.uniform(0.0, 0.01), 4)
    jitter_ms     = round(random.uniform(0.5, 3.0), 2)
    bgp_prefix_count = None
    bgp_flap_count   = None

    # --- ISP SCENARIO LOGIC ---

    # --- Check if the AI Agent already fixed this! ---
    is_fixed = _is_mitigated(device["id"], scenario)

    # --- ISP SCENARIO LOGIC ---
    if is_fixed:
        # The AI intervened! Force metrics to look healthy/mitigated.
        status = "HEALTHY"
        scenario = f"mitigated_{scenario}" 
        latency += random.randint(5, 15) # Slight penalty for backup path, but mostly normal
        

    elif scenario == "fiber_cut" and device["type"] == "link":
        status        = "CRITICAL"
        latency       = 9999
        packet_loss   = 100.0
        utilization   = 0.0
        throughput_gb = 0.0
        error_rate    = 1.0
        jitter_ms     = 0.0

    elif scenario == "bgp_flap" and device["type"] == "peering":
        status           = "DEGRADED"
        latency          += random.randint(100, 400)
        packet_loss      = round(random.uniform(2.0, 18.0), 3)
        jitter_ms        = round(random.uniform(20.0, 80.0), 2)
        bgp_flap_count   = random.randint(3, 25)
        bgp_prefix_count = random.randint(50, 200)  # reduced from normal ~800
        error_rate       = round(random.uniform(0.05, 0.20), 4)

    elif scenario == "ddos_attack" and device["type"] == "router":
        status        = "CRITICAL"
        latency       += random.randint(80, 350)
        utilization   = round(random.uniform(95.0, 100.0), 1)
        cpu_util      = round(random.uniform(88.0, 100.0), 1)
        cpu_temp      = random.randint(78, 96)
        packet_loss   = round(random.uniform(8.0, 30.0), 3)
        throughput_gb = round(random.uniform(18.0, 25.0), 2)  # maxed
        error_rate    = round(random.uniform(0.10, 0.40), 4)
        jitter_ms     = round(random.uniform(30.0, 120.0), 2)

    elif scenario == "hardware_degradation" and device["type"] == "switch":
        status      = "DEGRADED"
        cpu_temp    = random.randint(83, 99)
        cpu_util    = round(random.uniform(70.0, 90.0), 1)
        packet_loss = round(random.uniform(0.8, 6.0), 3)
        latency     += random.randint(15, 60)
        error_rate  = round(random.uniform(0.02, 0.08), 4)

    elif scenario == "peak_congestion" and device["type"] == "link":
        status        = "DEGRADED"
        utilization   = round(random.uniform(85.0, 99.0), 1)
        latency       += random.randint(25, 100)
        packet_loss   = round(random.uniform(0.3, 5.0), 3)
        throughput_gb = round(random.uniform(9.0, 12.0), 2)  # near capacity
        jitter_ms     = round(random.uniform(8.0, 35.0), 2)

    elif scenario == "dns_failure" and device["type"] == "dns":
        status      = "CRITICAL"
        latency     = random.randint(500, 3000)
        packet_loss = round(random.uniform(30.0, 80.0), 3)
        error_rate  = round(random.uniform(0.40, 0.95), 4)

    elif scenario == "link_flap" and device["type"] in ("link", "router"):
        # Intermittent brief outage
        status      = random.choice(["CRITICAL", "HEALTHY", "DEGRADED"])
        if status == "CRITICAL":
            latency     = random.randint(500, 2000)
            packet_loss = round(random.uniform(20.0, 60.0), 3)
        elif status == "DEGRADED":
            latency     += random.randint(50, 150)
            packet_loss = round(random.uniform(5.0, 15.0), 3)

    # Clamp utilization
    utilization = min(utilization, 100.0)
    cpu_util    = min(cpu_util, 100.0)

    payload = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "device_id":    device["id"],
        "device_type":  device["type"],
        "device_tier":  device["tier"],
        "region":       device["region"],
        "metrics": {
            "latency_ms":       max(0, latency),
            "packet_loss_pct":  packet_loss,
            "utilization_pct":  utilization,
            "cpu_temp_c":       cpu_temp,
            "cpu_util_pct":     cpu_util,
            "throughput_gbps":  throughput_gb,
            "error_rate":       error_rate,
            "jitter_ms":        jitter_ms,
        },
        "status":        status,
        "scenario_tag":  scenario,
        "tod_factor":    round(tod_factor, 2),
    }

    # Attach BGP-specific fields only for peering devices
    if device["type"] == "peering":
        payload["bgp"] = {
            "prefix_count":  bgp_prefix_count or random.randint(750, 850),
            "flap_count_1h": bgp_flap_count or 0,
            "as_path_len":   random.randint(2, 6),
        }

    return payload


def run_simulator(verbose: bool = True):
    """
    Runs the telemetry simulation loop forever (blocking).
    Safe to call from a background thread — e.g. server.py starts this
    in a daemon thread on startup so telemetry flows automatically
    without a separate process needing to be run manually.
    """
    if verbose:
        print("📡 Neurotech Telemetry Simulator — Enhanced v2")
        print("Scenarios: NORMAL, FIBER_CUT, BGP_FLAP, DDOS_ATTACK, "
              "HARDWARE_DEGRADATION, PEAK_CONGESTION, DNS_FAILURE, LINK_FLAP\n")

    scenarios = [
        "normal", "fiber_cut", "bgp_flap", "ddos_attack",
        "hardware_degradation", "peak_congestion", "dns_failure", "link_flap"
    ]
    # Realistic distribution: mostly normal, rare catastrophic events
    weights = [0.55, 0.04, 0.08, 0.02, 0.09, 0.12, 0.04, 0.06]

    COLOR = {
        "HEALTHY":  "\033[92m",
        "DEGRADED": "\033[93m",
        "CRITICAL": "\033[91m",
    }

    while True:
        try:
            current_mode = random.choices(scenarios, weights=weights, k=1)[0]

            # Burst mode for high-impact events
            burst = random.randint(10, 40) if current_mode in ("ddos_attack", "fiber_cut") else random.randint(1, 3)

            for _ in range(burst):
                t = generate_telemetry(scenario=current_mode)
                logger.info(json.dumps(t))

                if verbose:
                    c   = COLOR.get(t["status"], "\033[0m")
                    m   = t["metrics"]
                    bgp = f" | BGP flaps:{t['bgp']['flap_count_1h']}" if "bgp" in t else ""
                    print(
                        f"{c}[{t['scenario_tag']}] "
                        f"{t['device_id']:<30} | {t['status']:<8} | "
                        f"Lat:{m['latency_ms']:>5}ms | Loss:{m['packet_loss_pct']:>5}% | "
                        f"Util:{m['utilization_pct']:>5}% | CPU:{m['cpu_util_pct']:>5}% | "
                        f"Temp:{m['cpu_temp_c']:>2}°C{bgp}"
                        f"\033[0m"
                    )

            time.sleep(1.0)

        except KeyboardInterrupt:
            if verbose:
                print("\n⏹  Simulation stopped.")
            break
        except Exception as exc:
            # Never let the background thread die silently — log and keep going.
            if verbose:
                print(f"⚠ Simulator error (continuing): {exc}")
            time.sleep(1.0)


def main():
    run_simulator(verbose=True)


if __name__ == "__main__":
    main()