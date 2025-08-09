
import "./globals.css";
import React from "react";

export const metadata = { title: "DiCorner Demo", description: "Predictive churn & micro-persona insights" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <nav className="w-full bg-white border-b">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6">
            <a href="/" className="font-semibold">DiCorner Demo</a>
            <a href="/dashboard" className="text-sm">Dashboard</a>
            <a href="/model-evals" className="text-sm">Model Evals</a>
            <a href="/transparency" className="text-sm">Transparency</a>
            <a href="/what-if" className="text-sm">What‑if</a>
          <a href="/ethics-partnerships" className="text-sm">Ethics & Partnerships</a>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto p-6">{children}</main>
      </body>
    </html>
  );
}
