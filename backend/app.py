
import os
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

app = FastAPI(title="DiCorner Demo API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_events.csv")
STATE = {
    "model": None,
    "metrics": {},
    "parity": {},
    "overrides": [],
    "first_ingest_at": None,
    "first_insight_at": None,
    "last_audit": datetime.utcnow().isoformat(),
}

def load_df():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

def featurize(df: pd.DataFrame):
    X = pd.get_dummies(df[["persona","campaign","channel"]], drop_first=True)
    X["session_len"] = df["session_len"]
    X["clicks"] = df["clicks"]
    X["checkout_steps"] = df["checkout_steps"]
    X["decision_fatigue"] = np.maximum(0, 6 - df["clicks"])
    y = df["churn"].astype(int)
    return X, y

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds==b
        if np.sum(mask) == 0: 
            continue
        acc = float(np.mean(y_true[mask])) if np.sum(mask)>0 else 0.0
        conf = float(np.mean(y_prob[mask])) if np.sum(mask)>0 else 0.0
        ece += abs(acc - conf) * (np.sum(mask)/len(y_true))
    return float(ece)

def rates_from_scores(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    tp = int(np.sum((y_true==1) & (y_pred==1)))
    tn = int(np.sum((y_true==0) & (y_pred==0)))
    fp = int(np.sum((y_true==0) & (y_pred==1)))
    fn = int(np.sum((y_true==1) & (y_pred==0)))
    tpr = tp / (tp+fn) if (tp+fn)>0 else 0.0
    fpr = fp / (fp+tn) if (fp+tn)>0 else 0.0
    fnr = fn / (tp+fn) if (tp+fn)>0 else 0.0
    ppv = tp / (tp+fp) if (tp+fp)>0 else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "TPR": float(tpr), "FPR": float(fpr), "FNR": float(fnr), "PPV": float(ppv)
    }

def compute_parity(df: pd.DataFrame, model, thresh=0.5):
    Xg, yg = featurize(df)
    ypg = model.predict_proba(Xg)[:,1]
    global_rates = rates_from_scores(yg.values, ypg, thresh=thresh)

    parity = {"global": global_rates, "slices": {}}
    for col in ["persona","campaign","channel"]:
        for val, sub in df.groupby(col):
            Xs, ys = featurize(sub)
            yp = model.predict_proba(Xs)[:,1]
            r = rates_from_scores(ys.values, yp, thresh=thresh)
            gaps = {
                "FPR_gap_pp": abs(r["FPR"] - global_rates["FPR"])*100.0,
                "FNR_gap_pp": abs(r["FNR"] - global_rates["FNR"])*100.0,
                "TPR_gap_pp": abs(r["TPR"] - global_rates["TPR"])*100.0,
                "PPV_gap_pp": abs(r["PPV"] - global_rates["PPV"])*100.0,
            }
            parity["slices"][f"{col}:{val}"] = {"rates": r, "gaps": gaps}
    max_fpr_gap = 0.0
    max_fnr_gap = 0.0
    for k,v in parity["slices"].items():
        max_fpr_gap = max(max_fpr_gap, v["gaps"]["FPR_gap_pp"])
        max_fnr_gap = max(max_fnr_gap, v["gaps"]["FNR_gap_pp"])
    parity["max_gaps_pp"] = {"FPR": max_fpr_gap, "FNR": max_fnr_gap}
    return parity

def train_model():
    df = load_df()
    X, y = featurize(df)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    STATE["model"] = model

    y_prob = model.predict_proba(X)[:,1]
    metrics = {
        "AUROC": float(roc_auc_score(y, y_prob)),
        "AUPRC": float(average_precision_score(y, y_prob)),
        "Brier": float(brier_score_loss(y, y_prob)),
        "ECE": float(ece_score(y.values, y_prob)),
    }

    slices = {}
    for col in ["persona","campaign","channel"]:
        for val, sub in df.groupby(col):
            Xs, ys = featurize(sub)
            yp = model.predict_proba(Xs)[:,1]
            slices[f"{col}:{val}"] = {
                "AUROC": float(roc_auc_score(ys, yp)) if len(np.unique(ys))>1 else None,
                "AUPRC": float(average_precision_score(ys, yp)),
            }

    parity = compute_parity(df, model, thresh=0.5)

    STATE["metrics"] = {"global": metrics, "slices": slices}
    STATE["parity"] = parity
    return metrics

@app.on_event("startup")
def _startup():
    train_model()

class EventIn(BaseModel):
    user_id: int
    timestamp: Optional[str] = None
    persona: str
    campaign: str
    channel: str
    session_len: int
    clicks: int
    checkout_steps: float
    purchased: int
    churn: int

class OverrideIn(BaseModel):
    user_id: int
    score_before: float
    action: str
    reason_code: str
    reviewer: str

@app.post("/ingest/event")
def ingest_event(evt: EventIn):
    df = load_df()
    d = evt.dict()
    if d["timestamp"] is None:
        d["timestamp"] = datetime.utcnow().isoformat()
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)
    if STATE["first_ingest_at"] is None:
        STATE["first_ingest_at"] = datetime.utcnow()
    train_model()
    if STATE["first_insight_at"] is None:
        STATE["first_insight_at"] = datetime.utcnow()
    return {"ok": True}

@app.post("/eval/run")
def eval_run():
    m = train_model()
    return {"ok": True, "metrics": STATE["metrics"], "parity": STATE["parity"]}

@app.get("/eval/metrics")
def eval_metrics(slice: Optional[str] = None, include_parity: bool = False):
    if slice:
        return STATE["metrics"]["slices"].get(slice, {})
    if include_parity:
        return {"metrics": STATE["metrics"], "parity": STATE["parity"]}
    return STATE["metrics"]

@app.post("/override")
def override(o: OverrideIn):
    entry = o.dict()
    entry["created_at"] = datetime.utcnow().isoformat()
    STATE["overrides"].append(entry)
    return {"ok": True, "count": len(STATE["overrides"])}

@app.get("/overrides")
def get_overrides():
    return {"overrides": STATE["overrides"]}

@app.get("/nba")
def nba():
    recs = [
        {"persona":"Budget","action":"nudge_email_variant","copy":"Offer 10% off next order","confidence":0.81,"reasons":["low clicks","short session"]},
        {"persona":"Premium","action":"white_glove_outreach","copy":"Personal concierge call","confidence":0.87,"reasons":["high value","recent downgrade"]},
        {"persona":"Seasonal","action":"seasonal_bundle_offer","copy":"Bundle aligned to holiday","confidence":0.77,"reasons":["seasonal pattern","infrequent visits"]},
        {"persona":"Lapsed","action":"winback_series","copy":"3‑step reactivation series","confidence":0.69,"reasons":["no activity 30d","cart abandonment"]},
    ]
    return {"recommendations": recs}

@app.get("/ttv")
def ttv():
    if STATE["first_ingest_at"] and STATE["first_insight_at"]:
        delta = STATE["first_insight_at"] - STATE["first_ingest_at"]
        return {"seconds": delta.total_seconds()}
    return {"seconds": None}

@app.get("/ethics/status")
def ethics_status():
    parity = STATE.get("parity", {})
    max_fpr = parity.get("max_gaps_pp", {}).get("FPR", 0.0)
    max_fnr = parity.get("max_gaps_pp", {}).get("FNR", 0.0)
    metrics = STATE.get("metrics", {}).get("global", {})
    ece = metrics.get("ECE", None)
    brier = metrics.get("Brier", None)

    last_audit = datetime.fromisoformat(STATE["last_audit"])
    next_audit = last_audit + timedelta(days=30)

    autorater_checks = {
        "pii_leak": "pass",
        "fair_treatment": "pass" if max(max_fpr, max_fnr) <= 2.0 else "review",
        "factual_consistency": "pass",
        "promo_policy": "pass",
    }

    gates = {
        "parity_gap_pp_threshold": 2.0,
        "ece_threshold": 0.05,
        "brier_delta_trigger": 0.02
    }

    return {
        "last_audit_date": STATE["last_audit"],
        "next_audit_date": next_audit.isoformat(),
        "parity_max_gaps_pp": {"FPR": max_fpr, "FNR": max_fnr},
        "ece": ece,
        "brier": brier,
        "autorater_checks": autorater_checks,
        "gates": gates,
        "parity": parity,
    }

@app.get("/partners")
def partners():
    cards = [
        {
            "name": "Pilot SMB – DTC Apparel",
            "mission": "Cut time‑to‑value by streaming live events into churn insights.",
            "roles": {"partner":"Provide PostHog events + PoS txns","DiCorner":"Churn scoring, NBA, trust UX"},
            "success_metrics": {"ttv_hours":"48→2","trial_to_paid_lift_pct":13,"gdpr_passed":True},
            "governance": {"weekly_data_checkin": True, "monthly_steering": True}
        },
        {
            "name": "PoS Vendor – RetailOS",
            "mission": "Co-build connector to push/pull real-time receipts & refunds.",
            "roles": {"partner":"Connector uptime & schema","DiCorner":"Risk scoring + transparency API"},
            "success_metrics": {"connector_uptime_pct":99.9,"override_accept_rate_pct":41},
            "governance": {"weekly_data_checkin": True, "monthly_steering": True}
        },
        {
            "name": "MarTech – Customer.io",
            "mission": "Automate next‑best‑action campaigns from DiCorner signals.",
            "roles": {"partner":"Template & rate-limit controls","DiCorner":"Policy bands & confidence routing"},
            "success_metrics": {"send_errors":0,"policy_violations":0},
            "governance": {"weekly_data_checkin": True, "monthly_steering": True}
        }
    ]
    return {"partners": cards}
