
"use client";
import React, { useState } from "react";

export default function WhatIf() {
  const [threshold, setThreshold] = useState(0.75);
  const [applyRate, setApplyRate] = useState(0.58);

  const projectedLift = () => {
    // toy projection: higher threshold reduces apply rate but increases precision
    const precision = 0.6 + 0.4*(threshold-0.6);
    const acceptance = Math.max(0.3, applyRate - (threshold-0.75)*0.5);
    const lift = (precision*acceptance - 0.35) * 0.3; // arbitrary scaling
    return Math.max(0, lift);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">What‑if: Threshold & Policy Bands</h1>
      <div className="p-4 border rounded-xl bg-white">
        <label className="text-sm">Auto‑apply threshold: {threshold.toFixed(2)}</label>
        <input type="range" min="0.6" max="0.9" step="0.01" value={threshold} onChange={e=>setThreshold(parseFloat(e.target.value))} className="w-full"/>
      </div>
      <div className="p-4 border rounded-xl bg-white">
        <label className="text-sm">Current accept/apply rate: {applyRate.toFixed(2)}</label>
        <input type="range" min="0.3" max="0.9" step="0.01" value={applyRate} onChange={e=>setApplyRate(parseFloat(e.target.value))} className="w-full"/>
      </div>
      <div className="p-4 border rounded-xl bg-white">
        <div className="text-sm">Projected conversion lift:</div>
        <div className="text-2xl font-semibold">{(projectedLift()*100).toFixed(1)}%</div>
      </div>
    </div>
  );
}
