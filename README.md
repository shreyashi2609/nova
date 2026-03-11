# N.O.V.A.: Neurotech Operations and Virtual Agent

**N.O.V.A.** (Neurotech Operations and Virtual Agent) is an autonomous, self-healing network infrastructure agent. It replaces traditional, reactive Network Operations Center (NOC) monitoring with a proactive AI co-pilot that predicts congestion, diagnoses root causes, and executes remediation strategies with built-in Human-in-the-Loop (HITL) safety protocols.

---

## 1. Project Overview

### Core Vision
To eliminate the gap between anomaly detection and incident resolution. N.O.V.A. moves beyond simple "red-light" dashboards to a system that understands *why* a failure is happening and knows how to fix it before service levels are impacted.

### Key Features
* **Predictive Intelligence:** Foresees latency spikes before they cross critical thresholds.
* **Cognitive Root-Cause Analysis:** Uses Large Language Models (LLMs) to reason about telemetry logs like a senior engineer.
* **Autonomous Remediation:** Executes tools such as traffic rerouting, BGP resets, and ACL filtering.
* **Human-in-the-Loop (HITL):** A safety gate that pauses high-risk actions until an operator authorizes them via a real-time dashboard.
* **Incremental Learning:** Continuously retrains its models based on the success or failure of its own interventions.

### Tech Stack
* **Backend:** Python, LangGraph, LangChain, FastAPI, Scikit-Learn.
* **Intelligence:** Groq (Llama-4) for reasoning.
* **Frontend:** React, TypeScript, Chart.js, WebSockets.

---

## 2. Technical Architecture

The system operates on a continuous **Observe → Reason → Decide → Act → Learn** loop.

### The ML Pipeline (`backend/ml_models.py`)
N.O.V.A. utilizes a tri-layer machine learning approach to process telemetry data before it reaches the reasoning engine:
* **Anomaly Detection (Isolation Forest):** An unsupervised model that establishes a baseline for every device and flags statistical outliers.
* **Failure Classification (Gradient Boosting):** A supervised model trained to categorize failures into classes like `fiber_cut`, `ddos_attack`, or `bgp_flap`.
* **Latency Forecasting (Linear Regression):** A rolling regression model that predicts latency 5 steps into the future to provide early congestion warnings.

### The Agent Reasoning Engine (`backend/agent.py`)
Built using **LangGraph**, the agent coordinates the workflow through distinct nodes:
* **Observer:** Ingests telemetry and runs ML inference.
* **Reasoner:** An LLM node that analyzes the ML output and raw logs to form a technical hypothesis.
* **Decider:** Selects the appropriate remediation tool from a predefined catalog based on the failure type and risk level.
* **Sentry:** A human-authorization gate that interrupts the workflow for high-risk actions.
* **Executor:** Invokes real-world network tools and records the outcome.

---

## 3. Machine Learning Deep Dive

### Feature Engineering
N.O.V.A. extracts 8 critical metrics from every telemetry record: `latency_ms`, `packet_loss_pct`, `utilization_pct`, `cpu_temp_c`, `cpu_util_pct`, `throughput_gbps`, `error_rate`, and `jitter_ms`.

### Model Implementation Details
* **Isolation Forest:** Operates with a `contamination` factor of 0.08 (expecting 8% outliers). It requires at least 30 historical records (`min_train`) before it begins flagging anomalies for a specific device.
* **Gradient Boosting:** Uses 200 estimators and a depth of 4. It is bootstrapped with synthetic profiles but incrementally retrains every time it collects 50 new "confirmed" labels from the learning loop.
* **Linear Regression:** Fits a line across a rolling 20-tick window. It calculates the slope to label trends as RISING, FALLING, or STABLE, triggering an `early_warning` if the slope exceeds 2 and the predicted latency is high.

---

## 4. Operational Workflow

### Setup and Deployment
1.  **Backend:** Install dependencies via `pip install -r requirements.txt`.
2.  **Environment:** Set your `GROQ_API_KEY` in the `.env` file.
3.  **Execution:** Run `python server.py` to launch the FastAPI backend, which starts the background agent cycle and telemetry streamer.
4.  **Frontend:** Run `npm run dev` in the React directory to launch the N.O.V.A. Command Center.

### Human-in-the-Loop (HITL) Workflow
When N.O.V.A. identifies a high-risk failure (e.g., a BGP Flap requiring a reset):
1.  The agent pauses at the **Sentry Node**.
2.  The backend sends an `approval_required` signal via WebSocket.
3.  The React dashboard displays a critical overlay with the AI's reasoning and risk score.
4.  The operator clicks **Authorize** to proceed or **Deny** to revert to monitoring.

---
