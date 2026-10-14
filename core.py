
# Outputs (3 HTMLs):
# - context_graph.html        
# - risk_overlay.html         
# - governance_overlay.html   


import os
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Union

from pyvis.network import Network
from llm_engine import ask_chatgpt
from retrieval import retrieve_relevant_chunks


# Data model


@dataclass
class Node:
    id: str
    label: str
    type: str  # actor, component, external_system, requirement
    evidence: str = ""  # source sentence(s) this node was extracted from, when known


@dataclass
class Edge:
    source: str
    target: str
    label: str
    evidence: str = ""


@dataclass
class Risk:
    id: str
    description: str
    severity: str
    related_nodes: List[str]
    evidence: str = ""


@dataclass
class Assumption:
    id: str
    description: str
    related_nodes: List[str]
    evidence: str = ""


class ContextGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.risks: List[Risk] = []
        self.assumptions: List[Assumption] = []


GRAPH_VERSION = "v9.0-retrieval-and-multi-document-aggregation"



# Helpers


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _safe_id(label: str, node_type: str) -> str:
    return _sha256(f"{_norm(label)}|{node_type}")[:12]


def _find_existing_node_id(graph: ContextGraph, label: str, node_type: Optional[str] = None) -> Optional[str]:
    target = _norm(label)
    for nid, n in graph.nodes.items():
        if _norm(n.label) == target and (node_type is None or n.type == node_type):
            return nid
    return None


def ensure_node(graph: ContextGraph, label: str, node_type: str, node_id: Optional[str] = None, evidence: str = "") -> str:
    existing = _find_existing_node_id(graph, label, node_type)
    if existing:
        return existing

    nid = node_id or _safe_id(label, node_type)
    if nid in graph.nodes:
        nid = _safe_id(f"{label}::{nid}", node_type)

    graph.nodes[nid] = Node(id=nid, label=label, type=node_type, evidence=evidence)
    return nid


def add_edge(graph: ContextGraph, source_id: str, target_id: str, label: str, evidence: str = ""):
    if source_id not in graph.nodes or target_id not in graph.nodes:
        return
    graph.edges.append(Edge(source=source_id, target=target_id, label=label, evidence=evidence))


def _tooltip(label: str, evidence: str) -> str:
    """Hover-tooltip text for a pyvis node/edge: label alone, or label + source evidence when known."""
    if not evidence:
        return label
    return f"{label}\n\nEvidence: {evidence}"


# Relationship-label normalization (Phase 3): the LLM phrases the same kind
# of connection inconsistently across runs/domains (e.g. "streams_to" vs
# "sends_signals_to" vs "provides_alerts_to" -- exactly the gap the team's
# own Implementation report flagged as future work). This is a fixed lookup
# table, not a second LLM pass: deterministic, free, and consistent with how
# the rest of this file already layers deterministic post-processing on top
# of LLM output. Built from the labels actually observed across this
# project's eval runs (see docs/PROGRESS.md), not guessed. Labels not in the
# table are left unchanged rather than bucketed into a meaningless catch-all
# -- the eval harness measures what fraction get normalized (see run_evals.py).
RELATIONSHIP_LABEL_MAP = {
    # general data/control flow
    "transmits": "sends_to", "forwards": "sends_to", "streams": "sends_to",
    "provides": "sends_to", "provides_live_data": "sends_to", "reports_status": "sends_to",
    "hands_off": "sends_to", "sends_order": "sends_to", "sends_control_commands": "sends_to",
    "sends_schedule_updates": "sends_to", "sends_transaction_to": "sends_to",
    "sends_location_data": "sends_to", "sends_temperature_readings": "sends_to",
    "delivers": "sends_to", "receives": "sends_to", "transfers_funds_via": "sends_to",
    "sends_signals_to": "sends_to", "provides_alerts_to": "sends_to",
    # service invocation / RPC-like calls
    "calls": "calls", "queries": "calls", "initiates_transfer": "calls", "configures_device": "calls",
    # persistence
    "stores": "stores_in", "stores_route_history": "stores_in", "writes_to": "stores_in",
    "reads_writes": "stores_in",
    # event/alert propagation
    "triggers": "triggers", "triggers_alert": "triggers", "notifies": "triggers",
    # authentication
    "authenticates": "authenticates_via", "authenticates_via": "authenticates_via",
    # human oversight actions
    "reviews": "reviews", "acknowledges": "reviews", "flags_for_review": "reviews",
    "flags_for_review_on": "reviews", "flags_for_review_in": "reviews",
    # direct device/actuator control
    "controls": "controls",
}


def normalize_relationship_label(label: str) -> str:
    return RELATIONSHIP_LABEL_MAP.get(_norm(label).replace(" ", "_"), label)


def dedupe_edges(edges: List[Edge]) -> List[Edge]:
    seen: Set[Tuple[str, str, str]] = set()
    out: List[Edge] = []
    for e in edges:
        key = (e.source, e.target, e.label)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out



# LLM -- base graph 


def canonicalize_extraction(result: dict) -> ContextGraph:
    """
    Factored out of build_context_graph (Phase 5) so Phase 5's repair step
    can rebuild a ContextGraph from a corrected extraction dict without
    re-running retrieval/extraction or duplicating this logic.

    Canonicalize LLM entities by (label,type) so near-duplicates don't create orphan nodes.
    Rewrites relationship endpoints and risk/assumption related_nodes to canonical ids.
    """
    graph = ContextGraph()
    id_map: Dict[str, str] = {}

    # Entities
    for ent in result.get("entities", []):
        cid = ensure_node(graph, ent["label"], ent["type"], evidence=ent.get("evidence", ""))
        id_map[ent["id"]] = cid

    # Relationships
    for rel in result.get("relationships", []):
        s = id_map.get(rel["source_id"], rel["source_id"])
        t = id_map.get(rel["target_id"], rel["target_id"])
        add_edge(graph, s, t, normalize_relationship_label(rel["label"]), evidence=rel.get("evidence", ""))

    # Risks / assumptions
    for r in result.get("risks", []):
        related = [id_map.get(x, x) for x in r.get("related_nodes", [])]
        graph.risks.append(Risk(
            id=r.get("id", "R?"),
            description=r.get("description", ""),
            severity=r.get("severity", "medium"),
            related_nodes=related,
            evidence=r.get("evidence", "")
        ))

    for a in result.get("assumptions", []):
        related = [id_map.get(x, x) for x in a.get("related_nodes", [])]
        graph.assumptions.append(Assumption(
            id=a.get("id", "A?"),
            description=a.get("description", ""),
            related_nodes=related,
            evidence=a.get("evidence", "")
        ))

    graph.edges = dedupe_edges(graph.edges)
    return graph


def build_context_graph(doc_text: Union[str, List[Tuple[str, str]]], model: str = None) -> Tuple[dict, ContextGraph]:
    """
    doc_text: either a single input string (existing behavior, unchanged),
    or a list of (document_label, document_text) pairs for multi-document
    input (Phase 4). A plain string is wrapped as a single-document list
    internally, so every call site from before Phase 4 keeps working
    identically -- retrieval only activates once the combined word count
    passes retrieval.SIZE_THRESHOLD_WORDS (see retrieval.py), well above
    any of this project's existing single-document eval cases.

    model: optional override of llm_engine.MODEL_NAME, passed straight
    through to ask_chatgpt. None (the default) preserves existing behavior
    for every call site before Phase 6; Phase 6's model-comparison harness
    is the only caller that sets it.
    """
    documents = [("input", doc_text)] if isinstance(doc_text, str) else doc_text
    selected_text, retrieval_stats = retrieve_relevant_chunks(documents)

    result = ask_chatgpt(selected_text) if model is None else ask_chatgpt(selected_text, model=model)
    result["_retrieval_stats"] = retrieval_stats
    graph = canonicalize_extraction(result)
    return result, graph



# architecture realism 


def apply_base_improvements(graph: ContextGraph) -> None:
    patient = ensure_node(graph, "patient", "actor")
    nurses = ensure_node(graph, "nurses", "actor")
    doctors = ensure_node(graph, "doctors", "actor")
    it_admins = ensure_node(graph, "IT administrators", "actor")

    sensors = ensure_node(graph, "EEG/ECG sensors", "component")
    device = ensure_node(graph, "Patient Monitoring Device", "component")
    gateway = ensure_node(graph, "Streaming Gateway", "component")
    analytics = ensure_node(graph, "Real-Time Analytics Backend", "component")
    dashboard = ensure_node(graph, "Clinical Monitoring Dashboard", "component")
    notify = ensure_node(graph, "Notification Service", "component")

    ehr = ensure_node(graph, "Electronic Health Record (EHR) system", "external_system")
    idp = ensure_node(graph, "Hospital Identity Provider", "external_system")

    data_store = ensure_node(graph, "Time-Series Clinical Data Store", "component")
    audit = ensure_node(graph, "Audit Logging Service", "component")

    # Signal flow
    add_edge(graph, sensors, device, "measures")
    add_edge(graph, device, gateway, "forwards")
    add_edge(graph, gateway, analytics, "streams")

    # Analytics -> alerts -> clinicians
    add_edge(graph, analytics, notify, "triggers_alerts")
    add_edge(graph, notify, nurses, "notifies")
    add_edge(graph, notify, doctors, "notifies")
    add_edge(graph, nurses, dashboard, "reviews_alerts")
    add_edge(graph, doctors, dashboard, "reviews_alerts")
    add_edge(graph, nurses, dashboard, "acknowledges_alert")
    add_edge(graph, doctors, dashboard, "acknowledges_alert")

    # Auth + EHR boundary
    add_edge(graph, dashboard, idp, "authenticates_via")
    add_edge(graph, analytics, idp, "authenticates_via")
    add_edge(graph, analytics, ehr, "writes_summaries_to")
    add_edge(graph, device, ehr, "does_not_write_to") 

    # Admin
    add_edge(graph, it_admins, dashboard, "manages_config_and_RBAC")

    
    add_edge(graph, gateway, data_store, "stores_raw_stream")
    add_edge(graph, analytics, data_store, "stores_features_events")

    add_edge(graph, dashboard, audit, "logs_actions")
    add_edge(graph, notify, audit, "logs_delivery")
    add_edge(graph, analytics, audit, "logs_inference_alerts")

    graph.edges = dedupe_edges(graph.edges)



# UC3 parsing helpers


def extract_prefixed_lines(text: str, prefix: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        l = line.strip()
        if l.lower().startswith(prefix.lower()):
            out.append(l.split(":", 1)[1].strip())
    return out


def extract_functional_requirements_from_text(text: str) -> List[str]:
    t = _norm(text)
    frs: List[str] = []

    if "eeg" in t or "ecg" in t:
        frs.append("FR1: Acquire EEG/ECG signals continuously from wearable sensors.")
    if "patient monitoring device" in t:
        frs.append("FR2: Bedside device shall receive sensor signals and forward them via hospital network.")
    if "streaming gateway" in t or "streams" in t:
        frs.append("FR3: Streaming Gateway shall stream signals to the analytics backend with low latency.")
    if "analy" in t or "detect" in t:
        frs.append("FR4: Analytics backend shall detect abnormal events from EEG/ECG streams.")
    if "notification" in t or "alert" in t:
        frs.append("FR5: Notification service shall deliver alerts to clinicians with escalation support.")
    if "dashboard" in t:
        frs.append("FR6: Dashboard shall display alerts/trends and support acknowledgement/annotation.")
    if "ehr" in t:
        frs.append("FR7: Backend shall store summaries and alert metadata in the EHR; bedside devices shall not write to EHR.")
    if "identity provider" in t or "authenticate" in t:
        frs.append("FR8: Users and services shall authenticate via the Hospital Identity Provider.")
    if "rbac" in t or "role-based" in t:
        frs.append("FR9: Enforce RBAC for patient/nurse/doctor/IT administrator; IT admin manages roles/config.")

    if not frs:
        frs = [
            "FR1: Acquire physiological signals continuously.",
            "FR2: Stream signals to analytics backend.",
            "FR3: Detect abnormal events and notify clinicians.",
            "FR4: Store summaries in the EHR with enforced access control."
        ]

    #  preserve order
    seen = set()
    uniq = []
    for fr in frs:
        if fr not in seen:
            seen.add(fr)
            uniq.append(fr)
    return uniq



# UC3 enhancements: requirements, controls, ops, governance, mitigations


def apply_uc3_requirements_and_controls(graph: ContextGraph, input_text: str) -> None:
    # Key nodes
    sensors = ensure_node(graph, "EEG/ECG sensors", "component")
    device = ensure_node(graph, "Patient Monitoring Device", "component")
    gateway = ensure_node(graph, "Streaming Gateway", "component")
    analytics = ensure_node(graph, "Real-Time Analytics Backend", "component")
    dashboard = ensure_node(graph, "Clinical Monitoring Dashboard", "component")
    notify = ensure_node(graph, "Notification Service", "component")
    ehr = ensure_node(graph, "Electronic Health Record (EHR) system", "external_system")
    idp = ensure_node(graph, "Hospital Identity Provider", "external_system")
    data_store = ensure_node(graph, "Time-Series Clinical Data Store", "component")
    audit_service = ensure_node(graph, "Audit Logging Service", "component")

    # Requirements decomposition 
    frs = extract_functional_requirements_from_text(input_text)
    constraints = extract_prefixed_lines(input_text, "Constraint")
    assumptions = extract_prefixed_lines(input_text, "Assumption")
    risks_text = extract_prefixed_lines(input_text, "Risk")

    req_hub = ensure_node(graph, "Requirements & Compliance", "requirement")

    # NFRs: include user constraints + standard UC3 NFRs
    nfrs = [f"Constraint: {c}" for c in constraints] + [
        "NFR1: Protect medical data in transit (encryption + integrity).",
        "NFR2: Enforce least-privilege access and RBAC across roles.",
        "NFR3: Maintain audit logs for access, role changes, and alert acknowledgement.",
        "NFR4: Support continuous monitoring; handle network degradation gracefully.",
        "NFR5: Ensure safe alerting (escalation, acknowledgement tracking, timeouts)."
    ]

    fr_nodes: List[str] = []
    for fr in frs:
        nid = ensure_node(graph, fr, "requirement")
        fr_nodes.append(nid)
        add_edge(graph, req_hub, nid, "includes_FR")

    nfr_nodes: List[str] = []
    for nfr in nfrs:
        nid = ensure_node(graph, nfr, "requirement")
        nfr_nodes.append(nid)
        add_edge(graph, req_hub, nid, "includes_NFR")

    # Trace FRs to components 
    for fr_id in fr_nodes:
        lab = _norm(graph.nodes[fr_id].label)
        if "acquire" in lab or "sensor" in lab:
            add_edge(graph, fr_id, sensors, "satisfied_by")
        if "bedside" in lab or "forward" in lab:
            add_edge(graph, fr_id, device, "satisfied_by")
            add_edge(graph, fr_id, gateway, "satisfied_by")
        if "detect" in lab or "analytics" in lab:
            add_edge(graph, fr_id, analytics, "satisfied_by")
        if "notification" in lab or "alert" in lab:
            add_edge(graph, fr_id, notify, "satisfied_by")
        if "dashboard" in lab:
            add_edge(graph, fr_id, dashboard, "satisfied_by")
        if "ehr" in lab:
            add_edge(graph, fr_id, analytics, "satisfied_by")
            add_edge(graph, fr_id, ehr, "interface_to")
        if "identity" in lab or "authenticate" in lab:
            add_edge(graph, fr_id, idp, "depends_on")

    #  lifecycle 
    ops_hub = ensure_node(graph, "Operations & Lifecycle", "component")
    monitoring = ensure_node(graph, "Monitoring & Observability (metrics/logs/traces)", "component")
    incident = ensure_node(graph, "Incident Response (on-call/runbooks)", "component")
    maintenance = ensure_node(graph, "Maintenance & Change Management", "component")
    model_lifecycle = ensure_node(graph, "Model Lifecycle Management (validation/rollback)", "component")

    add_edge(graph, ops_hub, monitoring, "includes")
    add_edge(graph, ops_hub, incident, "includes")
    add_edge(graph, ops_hub, maintenance, "includes")
    add_edge(graph, ops_hub, model_lifecycle, "includes")

    add_edge(graph, monitoring, gateway, "monitors")
    add_edge(graph, monitoring, analytics, "monitors")
    add_edge(graph, monitoring, notify, "monitors")
    add_edge(graph, monitoring, dashboard, "monitors")

    add_edge(graph, incident, notify, "responds_to_failures")
    add_edge(graph, incident, analytics, "handles_incidents")
    add_edge(graph, maintenance, dashboard, "administers")
    add_edge(graph, maintenance, analytics, "deploys_updates_to")
    add_edge(graph, model_lifecycle, analytics, "manages_models_for")

    # Data governance (data classes + retention) 
    gov_hub = ensure_node(graph, "Data Governance", "requirement")

    data_raw = ensure_node(graph, "Data Class: Raw EEG/ECG Streams (sensitive medical data)", "requirement")
    data_derived = ensure_node(graph, "Data Class: Derived Features/Summaries (sensitive medical data)", "requirement")
    data_alerts = ensure_node(graph, "Data Class: Alerts & Metadata (sensitive medical data)", "requirement")
    data_audit = ensure_node(graph, "Data Class: Audit Logs (security-relevant)", "requirement")

    add_edge(graph, gov_hub, data_raw, "defines")
    add_edge(graph, gov_hub, data_derived, "defines")
    add_edge(graph, gov_hub, data_alerts, "defines")
    add_edge(graph, gov_hub, data_audit, "defines")

    retention_raw = ensure_node(graph, "Retention Control: Raw signals per policy (configurable duration)", "requirement")
    retention_derived = ensure_node(graph, "Retention Control: Summaries/metadata per policy (configurable duration)", "requirement")
    retention_audit = ensure_node(graph, "Retention Control: Audit logs per security policy (configurable duration)", "requirement")

    add_edge(graph, gov_hub, retention_raw, "includes")
    add_edge(graph, gov_hub, retention_derived, "includes")
    add_edge(graph, gov_hub, retention_audit, "includes")

    add_edge(graph, data_raw, sensors, "originates_from")
    add_edge(graph, data_raw, device, "handled_by")
    add_edge(graph, data_raw, gateway, "handled_by")
    add_edge(graph, data_raw, analytics, "processed_by")

    add_edge(graph, data_derived, analytics, "produced_by")
    add_edge(graph, data_derived, ehr, "stored_in")

    add_edge(graph, data_alerts, notify, "handled_by")
    add_edge(graph, data_alerts, dashboard, "displayed_in")
    add_edge(graph, data_alerts, ehr, "stored_in")

    add_edge(graph, data_audit, audit_service, "stored_in")

    add_edge(graph, retention_raw, data_store, "applies_to")
    add_edge(graph, retention_derived, ehr, "applies_to")
    add_edge(graph, retention_audit, audit_service, "applies_to")

    # Controls & mitigations 
    controls_hub = ensure_node(graph, "Controls & Mitigations", "requirement")

    c_tls = ensure_node(graph, "Control: Encrypt data in transit (TLS/mTLS where applicable)", "requirement")
    c_rbac = ensure_node(graph, "Control: RBAC + least privilege", "requirement")
    c_audit = ensure_node(graph, "Control: Audit logging + review", "requirement")
    c_escalate = ensure_node(graph, "Control: Escalation + acknowledgement tracking (timeouts/on-call)", "requirement")
    c_resilience = ensure_node(graph, "Control: Resilience (buffering/retry/failover)", "requirement")
    c_model = ensure_node(graph, "Control: Model monitoring/validation + rollback", "requirement")

    for c in [c_tls, c_rbac, c_audit, c_escalate, c_resilience, c_model]:
        add_edge(graph, controls_hub, c, "includes")

    # Apply controls to components 
    for node in [sensors, device, gateway, analytics, notify, dashboard, ehr, idp, data_store]:
        add_edge(graph, c_tls, node, "applies_to")
    for node in [dashboard, analytics, ehr]:
        add_edge(graph, c_rbac, node, "applies_to")
    for node in [audit_service, dashboard, analytics, notify, maintenance]:
        add_edge(graph, c_audit, node, "applies_to")
    for node in [notify, dashboard]:
        add_edge(graph, c_escalate, node, "applies_to")
    for node in [gateway, analytics, notify]:
        add_edge(graph, c_resilience, node, "applies_to")
    for node in [analytics, model_lifecycle, monitoring]:
        add_edge(graph, c_model, node, "applies_to")

    # Link NFRs -> controls (simple mapping)
    for nfr_id in nfr_nodes:
        n = _norm(graph.nodes[nfr_id].label)
        if "transit" in n or "encrypt" in n or "protected" in n:
            add_edge(graph, nfr_id, c_tls, "implemented_by")
        if "rbac" in n or "least" in n or "access" in n:
            add_edge(graph, nfr_id, c_rbac, "implemented_by")
        if "audit" in n:
            add_edge(graph, nfr_id, c_audit, "implemented_by")
        if "alert" in n or "escalation" in n or "timeouts" in n:
            add_edge(graph, nfr_id, c_escalate, "implemented_by")
        if "availability" in n or "reliability" in n or "network" in n:
            add_edge(graph, nfr_id, c_resilience, "implemented_by")

    #  Assumptions as requirement nodes (for governance overlay) 
    for a in assumptions:
        aid = ensure_node(graph, f"Assumption: {a}", "requirement")
        add_edge(graph, req_hub, aid, "includes_assumption")

    #  Risks: use LLM risks if present, otherwise parse from text 
    if not graph.risks and risks_text:
        for i, desc in enumerate(risks_text, start=1):
            graph.risks.append(Risk(f"R{i}", desc, "medium", [analytics, notify, dashboard]))

    # Create readable risk nodes + map to controls
    for r in graph.risks:
        risk_node = ensure_node(graph, f"Risk: {r.id} ({r.severity}) — {r.description}", "requirement")
        add_edge(graph, req_hub, risk_node, "includes_risk")

        d = _norm(r.description)
        if "false positive" in d or "alarm fatigue" in d:
            add_edge(graph, risk_node, c_escalate, "mitigated_by")
            add_edge(graph, risk_node, c_model, "mitigated_by")
        elif "false negative" in d or "missed" in d:
            add_edge(graph, risk_node, c_model, "mitigated_by")
            add_edge(graph, risk_node, c_escalate, "mitigated_by")
        elif "network" in d or "outage" in d or "delay" in d:
            add_edge(graph, risk_node, c_resilience, "mitigated_by")
        elif "privacy" in d or "access" in d or "insecure" in d:
            add_edge(graph, risk_node, c_tls, "mitigated_by")
            add_edge(graph, risk_node, c_rbac, "mitigated_by")
            add_edge(graph, risk_node, c_audit, "mitigated_by")
        elif "delayed response" in d or "response" in d:
            add_edge(graph, risk_node, c_escalate, "mitigated_by")
            add_edge(graph, risk_node, incident, "mitigated_by")
        else:
            add_edge(graph, risk_node, c_audit, "mitigated_by")

        # Limited traceability: connect to up to 3 related nodes
        related = [n for n in r.related_nodes if n in graph.nodes][:3]
        for n in related:
            add_edge(graph, risk_node, n, "risk_affects")

    graph.edges = dedupe_edges(graph.edges)



# Export (3 graphs)


def export_interactive_graphs(graph: ContextGraph, out_context_html: str, out_risk_html: str, out_governance_html: str):
    colors = {
        "component": "#4da6ff",
        "actor": "#66ff66",
        "external_system": "#ff6666",
        "requirement": "#ffaa00"
    }


    net1 = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    net1.set_options("""
    {
      "physics": { "enabled": false },
      "edges": {
        "arrows": { "to": { "enabled": true } },
        "smooth": { "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.2 },
        "font": { "size": 0 }
      },
      "nodes": { "shape": "box", "margin": 10, "font": { "size": 16 } }
    }
    """)

    context_types = {"actor", "component", "external_system"}
    allowed_ctx = {nid for nid, n in graph.nodes.items() if n.type in context_types}

    # Only count context-to-context edges, then keep connected nodes
    deg_ctx = {nid: 0 for nid in allowed_ctx}
    for e in graph.edges:
        if e.source in allowed_ctx and e.target in allowed_ctx:
            deg_ctx[e.source] += 1
            deg_ctx[e.target] += 1
    ctx_nodes = {nid for nid in allowed_ctx if deg_ctx.get(nid, 0) > 0}

    x_by_type = {"actor": -520, "component": 0, "external_system": 520}
    y_spacing = 180

    grouped: Dict[str, List[str]] = {"actor": [], "component": [], "external_system": []}
    for nid in ctx_nodes:
        grouped[graph.nodes[nid].type].append(nid)
    for t in grouped:
        grouped[t].sort(key=lambda nid: _norm(graph.nodes[nid].label))

    for t, nids in grouped.items():
        for idx, nid in enumerate(nids):
            n = graph.nodes[nid]
            net1.add_node(
                n.id,
                label=f"{n.label}\n({n.type})",
                title=_tooltip(n.label, n.evidence),
                color=colors.get(n.type, "#cccccc"),
                shape="box",
                margin=10,
                widthConstraint={"maximum": 340},
                x=x_by_type[t],
                y=idx * y_spacing,
                fixed=True
            )

    for e in graph.edges:
        if e.source in ctx_nodes and e.target in ctx_nodes:
            net1.add_edge(e.source, e.target, title=_tooltip(e.label, e.evidence), arrows="to")

    net1.write_html(out_context_html)

    # Utility for overlay nets (hierarchical, bounded)
    def _overlay_net() -> Network:
        net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
        net.set_options("""
        {
          "layout": { "hierarchical": {
            "enabled": true,
            "direction": "LR",
            "sortMethod": "directed",
            "nodeSpacing": 220,
            "levelSeparation": 280,
            "treeSpacing": 300
          }},
          "physics": { "enabled": false },
          "edges": {
            "arrows": { "to": { "enabled": true } },
            "smooth": { "type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.25 },
            "font": { "size": 0 }
          },
          "nodes": { "shape": "box", "margin": 10, "font": { "size": 16 } }
        }
        """)
        return net

    # Type levels for overlays (helps readability)
    level_by_type = {"actor": 0, "component": 1, "external_system": 2, "requirement": 3}

    def _add_node_overlay(net: Network, nid: str, label: str, title: str):
        n = graph.nodes[nid]
        net.add_node(
            nid,
            label=label,
            title=title,
            color=colors.get(n.type, "#cccccc"),
            shape="box",
            margin=10,
            widthConstraint={"maximum": 360},
            level=level_by_type.get(n.type, 1)
        )

   
   
    net2 = _overlay_net()

    # Add context nodes first
    for nid in ctx_nodes:
        n = graph.nodes[nid]
        _add_node_overlay(net2, nid, f"{n.label}\n({n.type})", _tooltip(n.label, n.evidence))

    # Add a risk hub + limited risk nodes
    hub = ensure_node(graph, "Risk Register", "requirement")
    _add_node_overlay(net2, hub, "Risk Register\n(requirement)", "Hub for risks")

    risk_req_nodes = [nid for nid, n in graph.nodes.items()
                      if n.type == "requirement" and _norm(n.label).startswith("risk:")]
    risk_req_nodes.sort(key=lambda nid: _norm(graph.nodes[nid].label))

    MAX_RISKS = 10
    MAX_LINKS = 3

    for rid in risk_req_nodes[:MAX_RISKS]:
        full = graph.nodes[rid].label
        short = full.split("—")[0].strip()
        _add_node_overlay(net2, rid, f"{short}\n(requirement)", full)
        net2.add_edge(rid, hub, title="registered_in", arrows="to")

        # Link to context nodes via existing edges
        links = []
        for e in graph.edges:
            if e.source == rid and e.target in ctx_nodes:
                links.append(e.target)
        for target in links[:MAX_LINKS]:
            net2.add_edge(rid, target, title="risk_affects", arrows="to")

    # Also include edges among shown nodes (context + selected risks + hub)
    shown2 = set(ctx_nodes) | set(risk_req_nodes[:MAX_RISKS]) | {hub}
    for e in graph.edges:
        if e.source in shown2 and e.target in shown2:
            net2.add_edge(e.source, e.target, title=_tooltip(e.label, e.evidence), arrows="to")

    net2.write_html(out_risk_html)


    net3 = _overlay_net()

    for nid in ctx_nodes:
        n = graph.nodes[nid]
        _add_node_overlay(net3, nid, f"{n.label}\n({n.type})", _tooltip(n.label, n.evidence))

    # Include requirement nodes by prefix/hubs
    include_prefixes = (
        "requirements & compliance",
        "controls & mitigations",
        "data governance",
        "fr",
        "nfr",
        "constraint:",
        "control:",
        "data class:",
        "retention control:",
        "assumption:",
        "risk:"
    )

    req_nodes = [nid for nid, n in graph.nodes.items()
                 if n.type == "requirement" and _norm(n.label).startswith(include_prefixes)]
    req_nodes.sort(key=lambda nid: _norm(graph.nodes[nid].label))

    # Include ops components 
    ops_keywords = (
        "operations & lifecycle",
        "monitoring & observability",
        "incident response",
        "maintenance & change management",
        "model lifecycle management"
    )
    ops_nodes = [nid for nid, n in graph.nodes.items()
                 if n.type == "component" and any(k in _norm(n.label) for k in ops_keywords)]
    ops_nodes.sort(key=lambda nid: _norm(graph.nodes[nid].label))

    MAX_REQ = 28
    MAX_OPS = 8

    for nid in req_nodes[:MAX_REQ]:
        full = graph.nodes[nid].label
        if len(full) > 80:
            short = full[:77] + "..."
        else:
            short = full
        _add_node_overlay(net3, nid, f"{short}\n(requirement)", full)

    for nid in ops_nodes[:MAX_OPS]:
        n = graph.nodes[nid]
        _add_node_overlay(net3, nid, f"{n.label}\n({n.type})", _tooltip(n.label, n.evidence))

    shown3 = set(ctx_nodes) | set(req_nodes[:MAX_REQ]) | set(ops_nodes[:MAX_OPS])

    for e in graph.edges:
        if e.source in shown3 and e.target in shown3:
            net3.add_edge(e.source, e.target, title=_tooltip(e.label, e.evidence), arrows="to")

    net3.write_html(out_governance_html)



# Pipeline


def run_pipeline(input_text: Union[str, List[Tuple[str, str]]], out_dir: str = "output") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # Cache key includes version so updates actually regenerate outputs. For
    # multi-document input, hash each (label, text) pair in order so the
    # same set of documents always maps to the same cache entry.
    if isinstance(input_text, str):
        cache_payload = input_text
    else:
        cache_payload = "\x1e".join(f"{label}\x1f{text}" for label, text in input_text)
    h = _sha256(cache_payload + "|" + GRAPH_VERSION)
    run_dir = os.path.join(out_dir, h[:12])
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "context_model.json")
    context_html_path = os.path.join(run_dir, "context_graph.html")
    risk_html_path = os.path.join(run_dir, "risk_overlay.html")
    governance_html_path = os.path.join(run_dir, "governance_overlay.html")

    # Legacy key expected by server route
    html_path = context_html_path

    # Cache hit
    if (os.path.exists(json_path)
            and os.path.exists(context_html_path)
            and os.path.exists(risk_html_path)
            and os.path.exists(governance_html_path)):
        return {
            "cached": True,
            "run_dir": run_dir,
            "json_path": json_path,
            "html_path": html_path,  # legacy
            "context_html_path": context_html_path,
            "risk_html_path": risk_html_path,
            "governance_html_path": governance_html_path
        }

    # 1) LLM extraction 
    result, graph = build_context_graph(input_text)

    # 2) Deterministic baseline 
    apply_base_improvements(graph)

    # 3) UC3 complete layer
    apply_uc3_requirements_and_controls(graph, input_text)

    # Save LLM JSON output (for evidence/debugging)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Export 3 diagrams
    export_interactive_graphs(graph, context_html_path, risk_html_path, governance_html_path)

    return {
        "cached": False,
        "run_dir": run_dir,
        "json_path": json_path,
        "html_path": html_path,  # legacy
        "context_html_path": context_html_path,
        "risk_html_path": risk_html_path,
        "governance_html_path": governance_html_path
    }
