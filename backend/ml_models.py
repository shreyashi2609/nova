import json
import math
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "latency_ms", "packet_loss_pct", "utilization_pct",
    "cpu_temp_c", "cpu_util_pct", "throughput_gbps",
    "error_rate", "jitter_ms",
]

FAILURE_CLASSES = [
    "normal", "fiber_cut", "bgp_flap", "ddos_attack",
    "hardware_degradation", "peak_congestion", "dns_failure", "link_flap",
]


def extract_features(record: dict) -> Optional[List[float]]:
    """Pull the 8 numeric features from a telemetry record."""
    m = record.get("metrics", {})
    try:
        return [
            float(m.get("latency_ms", 0)),
            float(m.get("packet_loss_pct", 0)),
            float(m.get("utilization_pct", 0)),
            float(m.get("cpu_temp_c", 0)),
            float(m.get("cpu_util_pct", 0)),
            float(m.get("throughput_gbps", 0)),
            float(m.get("error_rate", 0)),
            float(m.get("jitter_ms", 0)),
        ]
    except (TypeError, ValueError):
        return None

# Synthetic Training Data Generator

def _generate_synthetic_samples(n_per_class: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Create labelled synthetic training samples for bootstrap."""
    rng = np.random.default_rng(42)
    X, y = [], []

    profiles = {
        "normal":               dict(lat=(10,25),  loss=(0,0.1),   util=(30,65),  temp=(38,55),  cpu=(15,40),  tput=(2,8),    err=(0,0.01),  jit=(0.5,3)),
        "fiber_cut":            dict(lat=(9000,10000),loss=(99,100),util=(0,2),    temp=(38,55),  cpu=(5,20),   tput=(0,0.1),  err=(0.9,1.0), jit=(0,1)),
        "bgp_flap":             dict(lat=(150,600), loss=(2,18),   util=(40,70),  temp=(40,60),  cpu=(30,60),  tput=(1,5),    err=(0.05,0.2),jit=(20,80)),
        "ddos_attack":          dict(lat=(80,400),  loss=(8,30),   util=(95,100), temp=(78,96),  cpu=(88,100), tput=(18,25),  err=(0.10,0.4),jit=(30,120)),
        "hardware_degradation": dict(lat=(30,80),   loss=(0.8,6),  util=(40,70),  temp=(83,99),  cpu=(70,90),  tput=(1,6),    err=(0.02,0.08),jit=(2,15)),
        "peak_congestion":      dict(lat=(40,120),  loss=(0.3,5),  util=(85,99),  temp=(45,65),  cpu=(50,75),  tput=(9,12),   err=(0.01,0.05),jit=(8,35)),
        "dns_failure":          dict(lat=(500,3000),loss=(30,80),  util=(20,50),  temp=(38,55),  cpu=(20,45),  tput=(0,2),    err=(0.4,0.95),jit=(50,200)),
        "link_flap":            dict(lat=(100,2000),loss=(20,60),  util=(0,80),   temp=(38,60),  cpu=(20,55),  tput=(0,6),    err=(0.1,0.6), jit=(10,100)),
    }

    for label, p in profiles.items():
        for _ in range(n_per_class):
            row = [
                rng.uniform(*p["lat"]),
                rng.uniform(*p["loss"]),
                rng.uniform(*p["util"]),
                rng.uniform(*p["temp"]),
                rng.uniform(*p["cpu"]),
                rng.uniform(*p["tput"]),
                rng.uniform(*p["err"]),
                rng.uniform(*p["jit"]),
            ]
            X.append(row)
            y.append(label)

    return np.array(X), np.array(y)

# Per-Device Anomaly Detector  (IsolationForest)

class PerDeviceAnomalyDetector:
    def __init__(self, contamination: float = 0.08, min_train: int = 30):
        self.contamination = contamination
        self.min_train     = min_train
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._models:  Dict[str, IsolationForest] = {}
        self._scalers: Dict[str, StandardScaler]  = {}
        self._trained: Dict[str, bool] = defaultdict(bool)

    def update(self, record: dict) -> Optional[dict]:
        """
        Ingest one telemetry record.
        Returns anomaly result dict once the device has enough history, else None.
        """
        device_id = record.get("device_id", "unknown")
        features  = extract_features(record)
        if features is None:
            return None

        buf = self._history[device_id]
        buf.append(features)

        # Train / retrain model once we have enough baseline
        if len(buf) >= self.min_train and (len(buf) % 20 == 0 or not self._trained[device_id]):
            self._retrain(device_id, list(buf))

        if not self._trained[device_id]:
            return None

        scaler = self._scalers[device_id]
        model  = self._models[device_id]
        X      = scaler.transform([features])
        score  = model.decision_function(X)[0]   # negative = more anomalous
        pred   = model.predict(X)[0]              # -1 = anomaly, 1 = normal

        return {
            "device_id":    device_id,
            "is_anomaly":   pred == -1,
            "anomaly_score":round(float(score), 4),  # lower = more anomalous
            "severity":     self._score_to_severity(score),
        }

    def _retrain(self, device_id: str, data: List[List[float]]):
        X = np.array(data)
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)
        model  = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_s)
        self._scalers[device_id] = scaler
        self._models[device_id]  = model
        self._trained[device_id] = True

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score < -0.20:  return "CRITICAL"
        if score < -0.10:  return "HIGH"
        if score < -0.02:  return "MEDIUM"
        return "LOW"

# Failure Classifier  (GradientBoostingClassifier)

class FailureClassifier:
    def __init__(self):
        self.encoder = LabelEncoder().fit(FAILURE_CLASSES)
        self.scaler  = StandardScaler()
        self.model   = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        self._is_fit = False
        self._X_buf: List[List[float]] = []
        self._y_buf: List[str]         = []
        self._bootstrap()

    def _bootstrap(self):
        X_syn, y_syn = _generate_synthetic_samples(n_per_class=300)
        y_enc        = self.encoder.transform(y_syn)
        X_s          = self.scaler.fit_transform(X_syn)
        self.model.fit(X_s, y_enc)
        self._is_fit = True

    def predict(self, record: dict) -> Optional[dict]:
        if not self._is_fit:
            return None
        features = extract_features(record)
        if features is None:
            return None

        X    = self.scaler.transform([features])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        label      = self.encoder.inverse_transform([pred])[0]
        confidence = float(prob[pred])
        top2_idx   = np.argsort(prob)[-2:][::-1]
        top2       = [(self.encoder.inverse_transform([i])[0], round(float(prob[i]), 3)) for i in top2_idx]

        return {
            "predicted_class": label,
            "confidence":      round(confidence, 3),
            "top2_predictions": top2,
            "risk_score":      self._class_to_risk(label, confidence),
        }

    def feedback(self, record: dict, true_label: str):
        features = extract_features(record)
        if features and true_label in FAILURE_CLASSES:
            self._X_buf.append(features)
            self._y_buf.append(true_label)
            # Retrain once we have a meaningful batch
            if len(self._X_buf) >= 50:
                self._incremental_retrain()

    def _incremental_retrain(self):
        X_syn, y_syn  = _generate_synthetic_samples(n_per_class=200)
        X_new = np.array(self._X_buf)
        y_new = np.array(self._y_buf)
        X_all = np.vstack([X_syn, X_new])
        y_all = np.concatenate([y_syn, y_new])
        y_enc = self.encoder.transform(y_all)
        X_s   = self.scaler.fit_transform(X_all)
        self.model.fit(X_s, y_enc)
        self._X_buf.clear()
        self._y_buf.clear()

    @staticmethod
    def _class_to_risk(label: str, confidence: float) -> float:
        base = {
            "fiber_cut": 0.95, "ddos_attack": 0.90, "dns_failure": 0.80,
            "bgp_flap": 0.75,  "hardware_degradation": 0.65, "link_flap": 0.60,
            "peak_congestion": 0.45, "normal": 0.05,
        }.get(label, 0.3)
        return round(base * confidence, 3)

# Latency Forecaster  (rolling linear regression over time)

class LatencyForecaster:
    WINDOW = 20
    HORIZON = 5  # predict 5 ticks ahead

    def __init__(self):
        self._buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.WINDOW))
        self._models:  Dict[str, LinearRegression] = {}

    def update(self, record: dict) -> Optional[dict]:
        device_id = record.get("device_id", "unknown")
        m         = record.get("metrics", {})
        latency   = m.get("latency_ms", None)
        if latency is None:
            return None

        buf = self._buffers[device_id]
        buf.append(float(latency))

        if len(buf) < self.WINDOW:
            return None

        vals = np.array(list(buf))
        t    = np.arange(len(vals)).reshape(-1, 1)
        reg  = LinearRegression().fit(t, vals)
        self._models[device_id] = reg

        # Predict HORIZON steps ahead
        t_future  = np.array([[len(vals) + self.HORIZON]])
        predicted = float(reg.predict(t_future)[0])

        # Trend: slope of the regression line
        slope         = float(reg.coef_[0])
        trend_label   = "RISING" if slope > 2 else "FALLING" if slope < -2 else "STABLE"
        trend_warning = predicted > 200 and trend_label == "RISING"

        return {
            "device_id":         device_id,
            "current_latency_ms":round(vals[-1], 2),
            "predicted_latency_ms": round(predicted, 2),
            "slope_ms_per_tick": round(slope, 3),
            "trend":             trend_label,
            "early_warning":     trend_warning,
        }


# one interface for the agent to call

class NetworkMLIntelligence:
    def __init__(self):
        self.anomaly_detector  = PerDeviceAnomalyDetector()
        self.failure_classifier = FailureClassifier()
        self.latency_forecaster = LatencyForecaster()
        self._intervention_log: List[dict] = []  # for learning loop

    def ingest(self, record: dict) -> dict:
        anomaly    = self.anomaly_detector.update(record)
        prediction = self.failure_classifier.predict(record)
        forecast   = self.latency_forecaster.update(record)

        result = {
            "timestamp":  record.get("timestamp"),
            "device_id":  record.get("device_id"),
            "anomaly":    anomaly,
            "prediction": prediction,
            "forecast":   forecast,
        }

        # Aggregate risk score
        risk = 0.0
        if anomaly and anomaly["is_anomaly"]:
            risk = max(risk, {"LOW": 0.3, "MEDIUM": 0.55, "HIGH": 0.75, "CRITICAL": 0.95}.get(anomaly["severity"], 0.3))
        if prediction:
            risk = max(risk, prediction["risk_score"])
        if forecast and forecast.get("early_warning"):
            risk = max(risk, 0.50)

        result["aggregate_risk_score"] = round(risk, 3)
        return result

    def record_intervention(self, action: str, device_id: str, outcome: str, record: dict, true_label: str = None):
        """Log intervention outcome for the learning loop."""
        entry = {
            "timestamp":   datetime.utcnow().isoformat(),
            "action":      action,
            "device_id":   device_id,
            "outcome":     outcome,
            "true_label":  true_label,
        }
        self._intervention_log.append(entry)
        # Feed confirmed labels back to classifier
        if true_label and record:
            self.failure_classifier.feedback(record, true_label)

    def get_intervention_stats(self) -> dict:
        total     = len(self._intervention_log)
        successes = sum(1 for e in self._intervention_log if e["outcome"] == "SUCCESS")
        return {
            "total_interventions": total,
            "success_rate":        round(successes / total, 3) if total else 0.0,
            "recent":              self._intervention_log[-5:],
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "_main_":
    import sys
    sys.path.insert(0, ".")
    from backend.logger import generate_telemetry

    intel = NetworkMLIntelligence()
    print("Warming up ML models with 60 records...")
    for i in range(60):
        scenario = ["normal", "ddos_attack", "fiber_cut", "bgp_flap"][i % 4]
        rec = generate_telemetry(scenario=scenario)
        result = intel.ingest(rec)
        if result["aggregate_risk_score"] > 0:
            print(f"  [{result['device_id']:<30}] risk={result['aggregate_risk_score']:.2f} "
                  f"| pred={result['prediction']['predicted_class'] if result['prediction'] else 'n/a':<22}"
                  f"| anomaly={result['anomaly']['is_anomaly'] if result['anomaly'] else 'warming'}")

    print("\nStats:", intel.get_intervention_stats())
    print("ML models OK ✓")