
"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
import posthog from "posthog-js";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null);
  const [overrides, setOverrides] = useState<any[]>([]);
  const [ttv, setTtv] = useState<number | null>(null);

  useEffect(() => {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY || "", { api_host: "https://us.i.posthog.com" });
    const load = async () => {
      const m = await axios.get(`${API}/eval/metrics`);
      setMetrics(m.data);
      const o = await axios.get(`${API}/overrides`);
      setOverrides(o.data.overrides);
      const t = await axios.get(`${API}/ttv`);
      setTtv(t.data.seconds);
    };
    load();
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Main Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">AUROC</div>
          <div className="text-2xl font-semibold">{metrics?.global?.AUROC?.toFixed(3)}</div>
        </div>
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">AUPRC</div>
          <div className="text-2xl font-semibold">{metrics?.global?.AUPRC?.toFixed(3)}</div>
        </div>
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">ECE (calibration)</div>
          <div className="text-2xl font-semibold">{metrics?.global?.ECE?.toFixed(3)}</div>
        </div>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-2">Overrides</h3>
        <div className="text-sm">Count: {overrides.length}</div>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-2">Time‑to‑Value</h3>
        <div className="text-sm">{ttv ? `${Math.round(ttv)} seconds` : "—"}</div>
      </div>
    </div>
  );
}
