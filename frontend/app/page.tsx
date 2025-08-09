
"use client";
import React, { useEffect } from "react";
import posthog from "posthog-js";

export default function Home() {
  useEffect(() => {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY || "", { api_host: "https://us.i.posthog.com" });
    posthog.capture("landing_view");
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">DiCorner Demo</h1>
      <p className="text-sm">1‑Click insight for SMB retention: predictive churn, micro‑persona clustering, and next‑best‑actions with transparency & override.</p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <a className="p-4 border rounded-xl bg-white" href="/dashboard">
          <h3 className="font-semibold">Dashboard</h3><p className="text-sm">Trial→Paid, Churn Risk trend, Overrides</p>
        </a>
        <a className="p-4 border rounded-xl bg-white" href="/model-evals">
          <h3 className="font-semibold">Model Evals</h3><p className="text-sm">AUROC/AUPRC, Brier/ECE, slices</p>
        </a>
        <a className="p-4 border rounded-xl bg-white" href="/transparency">
          <h3 className="font-semibold">Transparency & Override</h3><p className="text-sm">Explain‑why, apply/edit/skip</p>
        </a>
      </div>
      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-2">Strategic & User‑Centric Layer</h3>
        <ul className="list-disc pl-5 text-sm space-y-1">
          <li>North‑star: Δ trial→paid, Δ M2 retention, ↓ analyst time</li>
          <li>Trust pack: transparency card, override, calibration (ECE/Brier), slice parity ≤ 2pp</li>
          <li>Responsible launch: canary→rollback, DPIA, governance, model registry</li>
        </ul>
      </div>
    </div>
  );
}
