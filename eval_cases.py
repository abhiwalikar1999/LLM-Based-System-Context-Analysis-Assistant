# eval_cases.py
#
# Phase 1 evaluation set: hand-labeled ground truth for the raw LLM extraction
# stage (llm_engine.ask_chatgpt -> core.build_context_graph), BEFORE the
# deterministic enrichment layers (apply_base_improvements /
# apply_uc3_requirements_and_controls) run.
#
# Why evaluate before enrichment, not the full pipeline:
#   apply_base_improvements unconditionally injects a hospital-specific
#   skeleton (patient / nurses / doctors / EEG/ECG sensors / ...) into every
#   graph, regardless of the input's actual domain (see core.py line ~171).
#   Running the full run_pipeline() against non-healthcare inputs would bolt
#   hallucinated-looking hospital nodes onto an e-commerce or logistics
#   graph, which would be meaningless to grade. So this eval set targets
#   exactly what varies with input text: the LLM's own extraction.
#
# Domains are deliberately varied (not just healthcare) to test whether the
# extraction prompt generalizes, rather than only working on the one
# scenario (system_description.txt) the deterministic layers were
# hand-tuned against.
#
# Ground truth format:
#   entities: list of {label, type} - type in {actor, component, external_system}
#     (requirement-type entities are excluded from ground truth: the prompt
#     can emit them, but they're not consistently expected, and the
#     deterministic layer is the primary source of requirement nodes anyway)
#   relationships: list of {source, target} - label text is NOT graded (see
#     run_evals.py for why); only whether the (source entity, target entity)
#     pair was found
#   risks / assumptions: list of short strings describing the expected risk
#     or assumption; graded by token-overlap coverage, not exact match

EVAL_CASES = [
    {
        "id": "healthcare_neurocardiowatch",
        "domain": "healthcare (reference scenario)",
        "input_text": """NeuroCardioWatch is a hospital monitoring solution for continuous acquisition and analysis of EEG and ECG signals.

A patient wears EEG/ECG sensors during monitoring. Sensors transmit signals to a bedside Patient Monitoring Device. The Patient Monitoring Device forwards signals to the Streaming Gateway over the hospital network, and the Streaming Gateway streams signals to the Real-Time Analytics Backend.

The Real-Time Analytics Backend analyzes EEG/ECG streams to detect abnormal events and generates alert events with associated metadata. It triggers the Notification Service, which delivers alerts to nurses and doctors. Nurses and doctors review alert details and trends in the Clinical Monitoring Dashboard and acknowledge alerts.

The Real-Time Analytics Backend stores analysis summaries and alert metadata in the Electronic Health Record (EHR) system. Users of the Clinical Monitoring Dashboard authenticate via the Hospital Identity Provider.

Risks: False positive alerts may lead to alarm fatigue. Network outages may interrupt monitoring and delay alerts.

Assumptions: The hospital network provides reliable connectivity. Clinicians respond to critical alerts within an acceptable time window.""",
        "ground_truth": {
            "entities": [
                {"label": "patient", "type": "actor"},
                {"label": "nurses", "type": "actor"},
                {"label": "doctors", "type": "actor"},
                {"label": "EEG/ECG sensors", "type": "component"},
                {"label": "Patient Monitoring Device", "type": "component"},
                {"label": "Streaming Gateway", "type": "component"},
                {"label": "Real-Time Analytics Backend", "type": "component"},
                {"label": "Notification Service", "type": "component"},
                {"label": "Clinical Monitoring Dashboard", "type": "component"},
                {"label": "Electronic Health Record", "type": "external_system"},
                {"label": "Hospital Identity Provider", "type": "external_system"},
            ],
            "relationships": [
                {"source": "EEG/ECG sensors", "target": "Patient Monitoring Device"},
                {"source": "Patient Monitoring Device", "target": "Streaming Gateway"},
                {"source": "Streaming Gateway", "target": "Real-Time Analytics Backend"},
                {"source": "Real-Time Analytics Backend", "target": "Notification Service"},
                {"source": "Real-Time Analytics Backend", "target": "Electronic Health Record"},
                {"source": "Notification Service", "target": "nurses"},
                {"source": "Notification Service", "target": "doctors"},
                {"source": "Clinical Monitoring Dashboard", "target": "Hospital Identity Provider"},
            ],
            "risks": [
                "false positive alerts cause alarm fatigue",
                "network outage interrupts monitoring and delays alerts",
            ],
            "assumptions": [
                "hospital network provides reliable connectivity",
                "clinicians respond to critical alerts in time",
            ],
        },
    },
    {
        "id": "ecommerce_bookstore",
        "domain": "e-commerce / retail",
        "input_text": """BookNook is an online bookstore platform. Customers browse and purchase books through the Web Storefront. When a customer checks out, the Web Storefront sends the order to the Order Service, which records it and calls the Payment Gateway (an external third-party processor) to charge the customer's card.

Once payment succeeds, the Order Service notifies the Inventory Service to reserve stock for the shipped items. The Inventory Service tracks stock levels in the Inventory Database. The Order Service also hands off fulfillment details to the Shipping Partner, an external logistics company, who delivers the package and reports delivery status back.

Store administrators manage the book catalog and pricing through the Admin Portal, which writes directly to the Inventory Database.

Risks: The Payment Gateway may become unavailable, blocking all checkouts. The Inventory Database may show stock as available when it has actually been oversold due to concurrent orders.

Assumptions: The Payment Gateway maintains PCI-DSS compliance. The Shipping Partner's API remains available during peak sales periods.""",
        "ground_truth": {
            "entities": [
                {"label": "customers", "type": "actor"},
                {"label": "store administrators", "type": "actor"},
                {"label": "Web Storefront", "type": "component"},
                {"label": "Order Service", "type": "component"},
                {"label": "Inventory Service", "type": "component"},
                {"label": "Inventory Database", "type": "component"},
                {"label": "Admin Portal", "type": "component"},
                {"label": "Payment Gateway", "type": "external_system"},
                {"label": "Shipping Partner", "type": "external_system"},
            ],
            "relationships": [
                {"source": "Web Storefront", "target": "Order Service"},
                {"source": "Order Service", "target": "Payment Gateway"},
                {"source": "Order Service", "target": "Inventory Service"},
                {"source": "Inventory Service", "target": "Inventory Database"},
                {"source": "Order Service", "target": "Shipping Partner"},
                {"source": "Admin Portal", "target": "Inventory Database"},
            ],
            "risks": [
                "payment gateway unavailability blocks checkout",
                "inventory oversold due to concurrent orders",
            ],
            "assumptions": [
                "payment gateway maintains PCI-DSS compliance",
                "shipping partner API stays available during peak periods",
            ],
        },
    },
    {
        "id": "smart_home_thermostat",
        "domain": "IoT / smart home",
        "input_text": """EcoTherm is a smart thermostat system for residential homes. Homeowners control their thermostat remotely through the EcoTherm Mobile App, which communicates with the Cloud Backend over the internet.

The Thermostat Device, installed in the home, continuously sends temperature readings to the Cloud Backend and receives updated schedules and target temperatures in return. The Thermostat Device directly controls the home's HVAC Unit based on the current target temperature.

The Cloud Backend queries the Weather API, an external third-party service, to adjust heating/cooling schedules based on forecasted weather. Installation technicians configure new devices using the Installer Portal, which connects to the Cloud Backend.

Risks: A Wi-Fi outage at the home prevents the Mobile App from controlling the Thermostat Device remotely. Inaccurate data from the Weather API can cause the system to schedule heating or cooling at the wrong times.

Assumptions: The home has a stable Wi-Fi connection. The HVAC Unit is compatible with the Thermostat Device's control protocol.""",
        "ground_truth": {
            "entities": [
                {"label": "homeowners", "type": "actor"},
                {"label": "installation technicians", "type": "actor"},
                {"label": "EcoTherm Mobile App", "type": "component"},
                {"label": "Cloud Backend", "type": "component"},
                {"label": "Thermostat Device", "type": "component"},
                {"label": "HVAC Unit", "type": "component"},
                {"label": "Installer Portal", "type": "component"},
                {"label": "Weather API", "type": "external_system"},
            ],
            "relationships": [
                {"source": "EcoTherm Mobile App", "target": "Cloud Backend"},
                {"source": "Thermostat Device", "target": "Cloud Backend"},
                {"source": "Thermostat Device", "target": "HVAC Unit"},
                {"source": "Cloud Backend", "target": "Weather API"},
                {"source": "Installer Portal", "target": "Cloud Backend"},
            ],
            "risks": [
                "wifi outage prevents remote control of thermostat",
                "inaccurate weather data causes wrong heating/cooling schedule",
            ],
            "assumptions": [
                "home has stable wifi connection",
                "HVAC unit is compatible with thermostat control protocol",
            ],
        },
    },
    {
        "id": "fintech_p2p_payments",
        "domain": "fintech / payments",
        "input_text": """PayLink is a peer-to-peer payment application. Users initiate money transfers to friends and family through the PayLink Mobile App, which sends transfer requests to the Transaction Service.

Before processing a transfer, the Transaction Service verifies the user's identity by calling the KYC Provider, an external identity verification service. Once verified, the Transaction Service moves funds between accounts via the Banking Rail, an external interbank settlement network.

The Fraud Detection Service monitors every transaction processed by the Transaction Service and flags suspicious activity. Compliance officers review flagged transactions through the Compliance Dashboard.

Risks: The Fraud Detection Service may fail to catch a fraudulent transfer (a false negative), allowing money to be stolen. An outage of the Banking Rail delays settlement of transfers.

Assumptions: The KYC Provider's identity data is accurate and up to date. Users have a verified bank account linked to their PayLink profile.""",
        "ground_truth": {
            "entities": [
                {"label": "users", "type": "actor"},
                {"label": "compliance officers", "type": "actor"},
                {"label": "PayLink Mobile App", "type": "component"},
                {"label": "Transaction Service", "type": "component"},
                {"label": "Fraud Detection Service", "type": "component"},
                {"label": "Compliance Dashboard", "type": "component"},
                {"label": "KYC Provider", "type": "external_system"},
                {"label": "Banking Rail", "type": "external_system"},
            ],
            "relationships": [
                {"source": "PayLink Mobile App", "target": "Transaction Service"},
                {"source": "Transaction Service", "target": "KYC Provider"},
                {"source": "Transaction Service", "target": "Banking Rail"},
                {"source": "Fraud Detection Service", "target": "Transaction Service"},
                {"source": "compliance officers", "target": "Compliance Dashboard"},
            ],
            "risks": [
                "fraud detection false negative allows fraudulent transfer",
                "banking rail outage delays settlement",
            ],
            "assumptions": [
                "KYC provider identity data is accurate and current",
                "users have a verified linked bank account",
            ],
        },
    },
    {
        "id": "logistics_fleet_tracking",
        "domain": "logistics / supply chain",
        "input_text": """FleetTrace is a delivery fleet tracking system. Drivers use the Driver App on their phones, which continuously sends GPS location data to the Tracking Backend.

The Tracking Backend stores each vehicle's route history in the Route Database. Dispatchers monitor live vehicle locations and delivery progress through the Dispatch Dashboard, which reads from the Tracking Backend.

When a delivery is near, the Tracking Backend sends an alert through the Notification Service to the customer. The Tracking Backend also queries the Traffic API, an external service, to calculate accurate estimated delivery times.

Risks: GPS signal loss in tunnels or dense urban areas causes inaccurate vehicle tracking. An outage of the Traffic API leads to incorrect estimated delivery times shown to customers.

Assumptions: Drivers' phones maintain continuous mobile data connectivity. All vehicles are equipped with GPS hardware compatible with the Driver App.""",
        "ground_truth": {
            "entities": [
                {"label": "drivers", "type": "actor"},
                {"label": "dispatchers", "type": "actor"},
                {"label": "customer", "type": "actor"},
                {"label": "Driver App", "type": "component"},
                {"label": "Tracking Backend", "type": "component"},
                {"label": "Route Database", "type": "component"},
                {"label": "Dispatch Dashboard", "type": "component"},
                {"label": "Notification Service", "type": "component"},
                {"label": "Traffic API", "type": "external_system"},
            ],
            "relationships": [
                {"source": "Driver App", "target": "Tracking Backend"},
                {"source": "Tracking Backend", "target": "Route Database"},
                {"source": "Dispatch Dashboard", "target": "Tracking Backend"},
                {"source": "Tracking Backend", "target": "Notification Service"},
                {"source": "Notification Service", "target": "customer"},
                {"source": "Tracking Backend", "target": "Traffic API"},
            ],
            "risks": [
                "GPS signal loss causes inaccurate tracking",
                "traffic API outage causes incorrect delivery time estimates",
            ],
            "assumptions": [
                "drivers phones have continuous data connectivity",
                "vehicles have GPS hardware compatible with driver app",
            ],
        },
    },
]
