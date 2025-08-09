
"use client";
import React, { useState } from "react";
import axios from "axios";
const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Transparency() {
  const [userId, setUserId] = useState(1);
  const [score, setScore] = useState(0.82);
  const [reason, setReason] = useState("creative_exception");
  const [reviewer, setReviewer] = useState("pm@dicorner.com");

  const submit = async () => {
    await axios.post(`${API}/override`, { user_id: userId, score_before: score, action: "override_apply", reason_code: reason, reviewer });
    alert("Override recorded");
  };

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Transparency Card & Override</h1>
      <div className="p-4 border rounded-xl bg-white">
        <pre className="text-xs overflow-auto">{JSON.stringify({
          answer: "Offer a 7‑day extension and re‑engagement email variant B.",
          confidence: 0.82,
          top_factors: ["recent_helpdesk_tickets","payment_retry_failures","drop_in_feature_use"],
          limitations: ["cold‑start user","limited CRM notes"],
          override_actions: ["apply","edit_copy","reject"],
          feedback_hook: "POST /override",
          why_this_model: "Best AUROC per $; calibrated (ECE 0.036); smallest worst‑slice gap."
        }, null, 2)}</pre>
      </div>
      <div className="p-4 border rounded-xl bg-white space-y-2">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <input className="border rounded px-2 py-1" type="number" value={userId} onChange={e=>setUserId(parseInt(e.target.value))} placeholder="user_id"/>
          <input className="border rounded px-2 py-1" type="number" value={score} step="0.01" onChange={e=>setScore(parseFloat(e.target.value))} placeholder="score_before"/>
          <input className="border rounded px-2 py-1" value={reason} onChange={e=>setReason(e.target.value)} placeholder="reason_code"/>
          <input className="border rounded px-2 py-1" value={reviewer} onChange={e=>setReviewer(e.target.value)} placeholder="reviewer"/>
        </div>
        <button onClick={submit} className="px-3 py-2 rounded bg-black text-white text-sm">Record Override</button>
      </div>
    </div>
  );
}
