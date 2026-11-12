"use client";

import { useEffect, useState, type ReactNode } from "react";
import { api, clearSession } from "@/lib/retrieval/api-source";
import { inputClass, primaryButtonClass } from "./retrieval-ui";

export function WorkspaceAccess({ children }: { children: ReactNode }) {
  const [authenticated, setAuthenticated] = useState(false);
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [expires, setExpires] = useState(0);
  useEffect(() => {
    const expired = () => { setAuthenticated(false); setExpires(0); };
    window.addEventListener("rag-session-expired", expired);
    return () => window.removeEventListener("rag-session-expired", expired);
  }, []);
  useEffect(() => {
    if (!expires) return;
    const timer = setTimeout(clearSession, Math.max(0, expires * 1000 - Date.now()));
    return () => clearTimeout(timer);
  }, [expires]);
  async function login() {
    setBusy(true); setError("");
    try {
      const session = await api.login(code);
      setCode(""); setExpires(session.expires_at); setAuthenticated(true);
    } catch (failure) { setError(failure instanceof Error ? failure.message : "Unable to sign in."); }
    finally { setBusy(false); }
  }
  if (!authenticated) return <main className="mx-auto mt-24 max-w-md rounded-xl border bg-white p-8 shadow-sm">
    <h1 className="text-2xl font-semibold">Retrieval Shield</h1>
    <p className="my-4 text-sm text-slate-600">Sign in to the shared capstone workspace. Documents are visible to approved workspace participants.</p>
    <form className="space-y-4" onSubmit={e => { e.preventDefault(); void login(); }}>
      <label className="block text-sm">Workspace access code<input type="password" autoComplete="current-password" required maxLength={256} className={`${inputClass} mt-2`} value={code} onChange={e => setCode(e.target.value)} /></label>
      <button disabled={busy} className={primaryButtonClass}>{busy ? "Signing in…" : "Sign in"}</button>
      {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
    </form>
  </main>;
  return <><div className="border-b bg-slate-50 px-5 py-2 text-right"><button className="text-xs underline" onClick={() => { void api.logout().catch(() => undefined); }}>Sign out</button></div>{children}</>;
}
