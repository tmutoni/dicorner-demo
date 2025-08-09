
"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";
const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function EthicsPartnerships() {
  const [ethics, setEthics] = useState<any>(null);
  const [partners, setPartners] = useState<any[]>([]);

  useEffect(() => {
    axios.get(`${API}/ethics/status`).then(r => setEthics(r.data));
    axios.get(`${API}/partners`).then(r => setPartners(r.data.partners));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">Ethics & Partnerships</h1>

      <div className="p-4 border rounded-xl bg-white space-y-2">
        <h2 className="font-semibold">Bias Mitigation Status</h2>
        {ethics && (
          <>
            <div className="text-sm">Last audit: {ethics.last_audit_date} | Next audit: {ethics.next_audit_date}</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
              {Object.entries(ethics.parity_gaps).map(([k,v]: any) => (
                <div key={k} className="p-3 border rounded">
                  <div className="text-sm font-medium">{k}</div>
                  <div className="text-xs">Max gap: {v.max_gap_pp} pp</div>
                  <div className={`text-xs ${v.status==="pass"?"text-green-600":"text-red-600"}`}>Status: {v.status}</div>
                </div>
              ))}
            </div>
            <div className="mt-3">
              <h3 className="text-sm font-medium">Autorater Checks</h3>
              <ul className="list-disc pl-5 text-xs">
                {Object.entries(ethics.autorater_checks).map(([k,v]: any) => (
                  <li key={k}>{k}: {v.status} (last run {v.last_run})</li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>

      <div className="p-4 border rounded-xl bg-white space-y-2">
        <h2 className="font-semibold">Partner Collaboration</h2>
        {partners.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {partners.map((p, i) => (
              <div key={i} className="p-3 border rounded text-sm space-y-1">
                <div className="font-medium">{p.name}</div>
                <div className="text-xs italic">{p.mission}</div>
                <div className="text-xs">Strengths: {p.strengths.join(", ")}</div>
                <div className="text-xs">Roles: Partner - {p.roles.Partner} | DiCorner - {p.roles.DiCorner}</div>
                <div className="text-xs">Results: {Object.entries(p.results).map(([rk, rv]) => `${rk}: ${rv}`).join(", ")}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
