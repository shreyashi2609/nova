import json
import os
import random
from datetime import datetime, timezone
from langchain_core.tools import tool

# ── Shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _rule_id() -> str:
    return f"rule_{random.getrandbits(20):05x}"

def _load(path: str, default) -> any:
    """Load JSON file; return default if missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def _save(path: str, data: any):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def _audit(tool_name: str, args: dict, result: str):
    """Append every tool call to the immutable audit log."""
    log = _load("audit_log.json", [])
    log.append({
        "id":        _rule_id(),
        "timestamp": _now(),
        "tool":      tool_name,
        "args":      args,
        "result":    result,
    })
    _save("audit_log.json", log)


# ── Tool 1: reroute_traffic ───────────────────────────────────────────────────

@tool
def reroute_traffic_tool(device_id: str, region: str, reason: str):
    """
    Reroutes traffic away from a degraded or failed device to its backup path.
    Use for: fiber_cut, link_flap, peak_congestion on a specific link.

    Args:
        device_id: The affected device ID (e.g., 'fiber_link_blr_chn').
        region:    The region of the affected device (e.g., 'South_Ring').
        reason:    Short description of why rerouting is needed.
    """
    routing_file = "network_routing.json"

    # 1. Load existing routing table (default: empty dict)
    routing = _load(routing_file, {})

    # 2. Build backup path entry
    backup_path = f"backup_{device_id}_{random.getrandbits(8):02x}"
    entry = {
        "device_id":   device_id,
        "region":      region,
        "primary":     device_id,
        "active_path": backup_path,
        "reason":      reason,
        "status":      "REROUTED",
        "timestamp":   _now(),
        "rule_id":     _rule_id(),
    }

    # 3. Update routing table keyed by device_id
    routing[device_id] = entry

    # 4. Save
    _save(routing_file, routing)

    result = (
        f"ACTION SUCCESS: Traffic rerouted away from '{device_id}' ({region}). "
        f"Active path → '{backup_path}'. Reason: {reason}."
    )
    _audit("reroute_traffic_tool", {"device_id": device_id, "region": region, "reason": reason}, result)
    return result


# ── Tool 2: apply_acl_filter ──────────────────────────────────────────────────

@tool
def apply_acl_filter_tool(device_id: str, region: str, threat_type: str, action_type: str = "RATE_LIMIT"):
    """
    Applies an ACL (Access Control List) filter on an edge device to block or
    rate-limit malicious or excessive traffic. Use for: ddos_attack.

    Args:
        device_id:   The router to apply the ACL on (e.g., 'edge_router_mumbai_01').
        region:      Region of the device (e.g., 'Mumbai').
        threat_type: Nature of the threat (e.g., 'DDoS', 'volumetric_flood').
        action_type: ACL action — 'RATE_LIMIT' (default) or 'DROP'.
    """
    acl_file = "acl_policy.json"

    # 1. Load existing ACL stack
    policies = _load(acl_file, [])

    # 2. Build new ACL rule
    new_rule = {
        "rule_id":    _rule_id(),
        "device_id":  device_id,
        "region":     region,
        "threat":     threat_type,
        "action":     action_type,
        "max_pps":    10000,          # packets-per-second cap
        "active":     True,
        "timestamp":  _now(),
    }

    # 3. Append to stack
    policies.append(new_rule)

    # 4. Save
    _save(acl_file, policies)

    result = (
        f"ACL APPLIED: {action_type} rule added on '{device_id}' ({region}) "
        f"for threat '{threat_type}'. Rule ID: {new_rule['rule_id']}. "
        f"Total active ACL rules: {len(policies)}."
    )
    _audit("apply_acl_filter_tool", {"device_id": device_id, "region": region, "threat_type": threat_type}, result)
    return result


# ── Tool 3: bgp_reset_tool ────────────────────────────────────────────────────

@tool
def bgp_reset_tool(peer_id: str, region: str, reason: str, reset_type: str = "soft"):
    """
    Resets a BGP peering session to restore stable routing after a BGP flap
    or extreme latency on a peering link.

    Args:
        peer_id:    The BGP peer device ID (e.g., 'bgp_peer_as1234').
        region:     Region of the peer (e.g., 'Global').
        reason:     Why the reset is being triggered.
        reset_type: 'soft' (safe, just refreshes prefix table) or
                    'hard' (full session teardown — use only if soft fails).
    """
    bgp_file = "bgp_state.json"

    # 1. Load BGP session registry
    sessions = _load(bgp_file, {})

    # 2. Update or create session entry
    sessions[peer_id] = {
        "peer_id":    peer_id,
        "region":     region,
        "last_reset": _now(),
        "reset_type": reset_type,
        "reason":     reason,
        "status":     "RESETTING",
        "rule_id":    _rule_id(),
    }

    # 3. Save
    _save(bgp_file, sessions)

    result = (
        f"BGP RESET INITIATED: {reset_type.upper()} reset on peer '{peer_id}' ({region}). "
        f"Prefix table will refresh within 30–90 seconds. Reason: {reason}."
    )
    _audit("bgp_reset_tool", {"peer_id": peer_id, "region": region, "reset_type": reset_type}, result)
    return result


# ── Tool 4: rate_limit_edge_tool ──────────────────────────────────────────────

@tool
def rate_limit_edge_tool(device_id: str, region: str, limit_gbps: float, reason: str):
    """
    Applies a throughput rate-limit cap on an edge router as a pre-emptive
    DDoS mitigation measure before full ACL deployment.

    Args:
        device_id:  The edge router to cap (e.g., 'edge_router_delhi_01').
        region:     Region of the device (e.g., 'Delhi').
        limit_gbps: Maximum throughput to allow in Gbps (e.g., 5.0).
        reason:     Reason for applying the cap.
    """
    acl_file = "acl_policy.json"

    # 1. Load existing ACL / rate-limit stack (shared file with apply_acl)
    policies = _load(acl_file, [])

    # 2. Build rate-limit rule
    new_rule = {
        "rule_id":    _rule_id(),
        "device_id":  device_id,
        "region":     region,
        "action":     "RATE_LIMIT_EDGE",
        "limit_gbps": limit_gbps,
        "reason":     reason,
        "active":     True,
        "timestamp":  _now(),
    }

    # 3. Append
    policies.append(new_rule)

    # 4. Save
    _save(acl_file, policies)

    result = (
        f"RATE LIMIT SET: Edge router '{device_id}' ({region}) capped at {limit_gbps} Gbps. "
        f"Rule ID: {new_rule['rule_id']}. Reason: {reason}."
    )
    _audit("rate_limit_edge_tool", {"device_id": device_id, "region": region, "limit_gbps": limit_gbps}, result)
    return result


# ── Tool 5: capacity_reallocation_tool ────────────────────────────────────────

@tool
def capacity_reallocation_tool(congested_link: str, region: str, target_links: str, reason: str):
    """
    Redistributes traffic load from a congested link to underutilized links
    in the same region. Use for: peak_congestion.

    Args:
        congested_link: The overloaded link device ID (e.g., 'fiber_link_mum_del').
        region:         Region of the congested link (e.g., 'West_North').
        target_links:   Comma-separated list of relief link IDs to shift load onto.
        reason:         Reason for reallocation.
    """
    capacity_file = "capacity_plan.json"

    # 1. Load existing capacity plan
    plan = _load(capacity_file, {})

    # 2. Build reallocation entry
    targets = [t.strip() for t in target_links.split(",") if t.strip()]
    entry = {
        "rule_id":       _rule_id(),
        "congested_link":congested_link,
        "region":        region,
        "relief_links":  targets,
        "load_split_pct":round(100 / max(len(targets), 1), 1),
        "reason":        reason,
        "status":        "ACTIVE",
        "timestamp":     _now(),
    }

    # 3. Key by congested link
    plan[congested_link] = entry

    # 4. Save
    _save(capacity_file, plan)

    result = (
        f"CAPACITY REALLOCATED: Load shifted from '{congested_link}' ({region}) "
        f"to relief links {targets}. Each relief link absorbs ~{entry['load_split_pct']}% "
        f"of diverted traffic. Reason: {reason}."
    )
    _audit("capacity_reallocation_tool",
           {"congested_link": congested_link, "region": region, "target_links": target_links}, result)
    return result


# ── Tool 6: config_rollback_tool ──────────────────────────────────────────────

@tool
def config_rollback_tool(device_id: str, region: str, snapshot_id: str, reason: str):
    """
    Rolls back a device's configuration to a known-good snapshot.
    HIGH RISK — irreversible without a new snapshot. Use when misconfiguration
    is confirmed as the root cause.

    Args:
        device_id:   The device to roll back (e.g., 'core_switch_delhi_05').
        region:      Region of the device (e.g., 'Delhi').
        snapshot_id: ID of the snapshot to restore. Use 'latest' for the
                     most recent known-good config.
        reason:      Confirmed root cause justifying the rollback.
    """
    routing_file = "network_routing.json"

    # 1. Load routing / config state
    state = _load(routing_file, {})

    # 2. Record rollback
    resolved_snapshot = snapshot_id if snapshot_id != "latest" else f"snap_{_rule_id()}"
    entry = {
        "rule_id":   _rule_id(),
        "device_id": device_id,
        "region":    region,
        "action":    "CONFIG_ROLLBACK",
        "snapshot":  resolved_snapshot,
        "reason":    reason,
        "status":    "ROLLED_BACK",
        "timestamp": _now(),
    }
    state[f"rollback_{device_id}"] = entry

    # 3. Save
    _save(routing_file, state)

    result = (
        f"CONFIG ROLLBACK COMPLETE: '{device_id}' ({region}) restored to snapshot "
        f"'{resolved_snapshot}'. Device will reload in ~45 seconds. Reason: {reason}."
    )
    _audit("config_rollback_tool",
           {"device_id": device_id, "region": region, "snapshot_id": snapshot_id}, result)
    return result


# ── Tool 7: dns_failover_tool ─────────────────────────────────────────────────

@tool
def dns_failover_tool(resolver_id: str, region: str, backup_resolver: str, reason: str):
    """
    Fails over DNS resolution from a degraded primary resolver to a healthy
    secondary resolver. Use for: dns_failure.

    Args:
        resolver_id:      The failing DNS resolver (e.g., 'dns_resolver_mumbai').
        region:           Region of the resolver (e.g., 'Mumbai').
        backup_resolver:  Target backup resolver to promote (e.g., 'dns_resolver_delhi').
        reason:           Why the failover is being triggered.
    """
    dns_file = "dns_config.json"

    # 1. Load DNS config
    config = _load(dns_file, {})

    # 2. Update resolver assignment
    config[region] = {
        "rule_id":          _rule_id(),
        "primary_resolver": resolver_id,
        "active_resolver":  backup_resolver,
        "status":           "FAILED_OVER",
        "reason":           reason,
        "timestamp":        _now(),
    }

    # 3. Save
    _save(dns_file, config)

    result = (
        f"DNS FAILOVER COMPLETE: Region '{region}' now resolving via '{backup_resolver}'. "
        f"Failed resolver: '{resolver_id}'. Reason: {reason}."
    )
    _audit("dns_failover_tool",
           {"resolver_id": resolver_id, "region": region, "backup_resolver": backup_resolver}, result)
    return result


# ── Tool 8: escalate_to_engineer_tool ─────────────────────────────────────────

@tool
def escalate_to_engineer_tool(device_id: str, region: str, severity: str, hypothesis: str, recommended_action: str):
    """
    Creates a PagerDuty-style incident and notifies the on-call engineer
    (Manan Shah). Use for: hardware_degradation, thermal issues, any scenario
    where autonomous action would be unsafe or insufficient.

    Args:
        device_id:          The device requiring human attention.
        region:             Region of the device.
        severity:           Incident severity — 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.
        hypothesis:         The agent's root-cause hypothesis to give the engineer context.
        recommended_action: What the agent recommends the engineer should do.
    """
    incident_file = "incident_log.json"

    # 1. Load incident log
    incidents = _load(incident_file, [])

    # 2. Create incident record
    incident_id = f"INC-{random.randint(10000, 99999)}"
    new_incident = {
        "incident_id":        incident_id,
        "rule_id":            _rule_id(),
        "device_id":          device_id,
        "region":             region,
        "severity":           severity,
        "hypothesis":         hypothesis,
        "recommended_action": recommended_action,
        "status":             "OPEN",
        "assigned_to":        "Manan Shah (on-call)",
        "notified_via":       ["PagerDuty", "Slack #noc-alerts", "SMS"],
        "timestamp":          _now(),
    }

    # 3. Append incident
    incidents.append(new_incident)

    # 4. Save
    _save(incident_file, incidents)

    result = (
        f"ESCALATION SENT: Incident {incident_id} created for '{device_id}' ({region}). "
        f"Severity: {severity}. Assigned to: Manan Shah (on-call). "
        f"Notified via PagerDuty + Slack #noc-alerts + SMS. "
        f"Hypothesis: {hypothesis}. Recommended: {recommended_action}."
    )
    _audit("escalate_to_engineer_tool",
           {"device_id": device_id, "region": region, "severity": severity}, result)
    return result


# ── Tool registry (imported by agent.py) ─────────────────────────────────────

ALL_TOOLS = [
    reroute_traffic_tool,
    apply_acl_filter_tool,
    bgp_reset_tool,
    rate_limit_edge_tool,
    capacity_reallocation_tool,
    config_rollback_tool,
    dns_failover_tool,
    escalate_to_engineer_tool,
]

TOOL_MAP = {t.name if hasattr(t, "name") else t.__name__: t for t in ALL_TOOLS}
