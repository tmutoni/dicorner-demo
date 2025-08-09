
"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function ModelEvals() {
  const [metrics, setMetrics] = useState<any>(null);
  const [sliceKey, setSliceKey] = useState<string>("persona:Budget");
  const [sliceMetrics, setSliceMetrics] = useState<any>(null);

  useEffect(() => {
    axios.get(`${API}/eval/metrics`).then(r => setMetrics(r.data));
  }, []);

  useEffect(() => {
    axios.get(`${API}/eval/metrics`, { params: { slice: sliceKey } }).then(r => setSliceMetrics(r.data));
  }, [sliceKey]);

  const slices = metrics ? Object.keys(metrics.slices || {}) : [];

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Model Evaluations (GRACE highlights)</h1>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 border rounded-xl bg-white"><div className="text-sm">AUROC</div><div className="text-2xl font-semibold">{metrics?.global?.AUROC?.toFixed(3)}</div></div>
        <div className="p-4 border rounded-xl bg-white"><div className="text-sm">AUPRC</div><div className="text-2xl font-semibold">{metrics?.global?.AUPRC?.toFixed(3)}</div></div>
        <div className="p-4 border rounded-xl bg-white"><div className="text-sm">Brier</div><div className="text-2xl font-semibold">{metrics?.global?.Brier?.toFixed(3)}</div></div>
        <div className="p-4 border rounded-xl bg-white"><div className="text-sm">ECE</div><div className="text-2xl font-semibold">{metrics?.global?.ECE?.toFixed(3)}</div></div>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <div className="flex items-center gap-3">
          <label className="text-sm">Slice</label>
          <select className="border rounded px-2 py-1" value={sliceKey} onChange={e=>setSliceKey(e.target.value)}>
            {slices.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="mt-3 text-sm">AUROC: {sliceMetrics?.AUROC ? sliceMetrics.AUROC.toFixed(3) : "—"} | AUPRC: {sliceMetrics?.AUPRC ? sliceMetrics.AUPRC.toFixed(3) : "—"}</div>
      </div>
    </div>
  );
}
