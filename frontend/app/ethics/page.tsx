
"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type ParitySlice = {
  rates: { TPR:number; FPR:number; FNR:number; PPV:number };
  gaps: { FPR_gap_pp:number; FNR_gap_pp:number; TPR_gap_pp:number; PPV_gap_pp:number };
};

export default function Ethics() {
  const [status, setStatus] = useState<any>(null);
  const [partners, setPartners] = useState<any[]>([]);
  const [keys, setKeys] = useState<string[]>([]);

  useEffect(() => {
    const load = async () => {
      const s = await axios.get(`${API}/ethics/status`);
      setStatus(s.data);
      const k = Object.keys(s.data?.parity?.slices || {});
      setKeys(k);
      const p = await axios.get(`${API}/partners`);
      setPartners(p.data.partners || []);
    };
    load();
  }, []);

  const fmt = (v:number|undefined|null) => v==null ? "—" : v.toFixed(3);

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Ethics & Partnerships</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">ECE</div>
          <div className="text-2xl font-semibold">{fmt(status?.ece)}</div>
        </div>
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">Brier</div>
          <div className="text-2xl font-semibold">{fmt(status?.brier)}</div>
        </div>
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">Max FPR Gap (pp)</div>
          <div className="text-2xl font-semibold">{status?.parity_max_gaps_pp?.FPR?.toFixed(2)}</div>
        </div>
        <div className="p-4 border rounded-xl bg-white">
          <div className="text-sm text-gray-600">Max FNR Gap (pp)</div>
          <div className="text-2xl font-semibold">{status?.parity_max_gaps_pp?.FNR?.toFixed(2)}</div>
        </div>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-2">Autorater & Safety Gates</h3>
        <pre className="text-xs overflow-auto">
{`PII: ${status?.autorater_checks?.pii_leak}
Fair Treatment: ${status?.autorater_checks?.fair_treatment}
Factual Consistency: ${status?.autorater_checks?.factual_consistency}
Promo Policy: ${status?.autorater_checks?.promo_policy}`}
        </pre>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-3">Slice Parity (gaps in percentage points)</h3>
        <div className="overflow-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr>
                <th className="text-left p-2 border-b">Slice</th>
                <th className="text-left p-2 border-b">FPR gap</th>
                <th className="text-left p-2 border-b">FNR gap</th>
                <th className="text-left p-2 border-b">TPR gap</th>
                <th className="text-left p-2 border-b">PPV gap</th>
              </tr>
            </thead>
            <tbody>
              {keys.map(k => {
                const s: ParitySlice = status?.parity?.slices?.[k];
                return (
                  <tr key={k}>
                    <td className="p-2 border-b">{k}</td>
                    <td className="p-2 border-b">{s?.gaps?.FPR_gap_pp.toFixed(2)}</td>
                    <td className="p-2 border-b">{s?.gaps?.FNR_gap_pp.toFixed(2)}</td>
                    <td className="p-2 border-b">{s?.gaps?.TPR_gap_pp.toFixed(2)}</td>
                    <td className="p-2 border-b">{s?.gaps?.PPV_gap_pp.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="p-4 border rounded-xl bg-white">
        <h3 className="font-semibold mb-3">Partner Co‑Builder Cards</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {partners.map((p:any, i:number)=>(
            <div key={i} className="border rounded-xl p-3">
              <div className="font-semibold">{p.name}</div>
              <div className="text-xs text-gray-600 mb-2">{p.mission}</div>
              <div className="text-xs"><span className="font-semibold">Roles</span>: {p.roles.partner} • {p.roles.DiCorner}</div>
              <div className="text-xs mt-1"><span className="font-semibold">Success</span>: {Object.entries(p.success_metrics).map(([k,v])=>`${k}: ${v}`).join(" | ")}</div>
              <div className="text-xs mt-1"><span className="font-semibold">Governance</span>: weekly data check‑in ✓ • monthly steering ✓</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
