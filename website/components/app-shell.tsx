"use client";

import { useState, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const groups = [
  { label: "Overview", pages: [["/dashboard", "Dashboard"]] },
  { label: "Workspace", pages: [["/search", "Search"], ["/documents", "Documents"]] },
  { label: "Security", pages: [["/security", "Security Review"], ["/activity", "Activity"], ["/evaluation", "Evaluation"]] },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);
  const title = groups.flatMap(group => group.pages).find(([href]) => pathname === href)?.[1] ?? "Retrieval Shield";
  return <div className="min-h-screen lg:grid lg:grid-cols-[224px_minmax(0,1fr)]">
    <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-white focus:p-3">Skip to content</a>
    <aside className="bg-slate-950 text-slate-200 lg:sticky lg:top-0 lg:h-screen lg:overflow-y-auto">
      <div className="flex items-center justify-between gap-3 px-5 py-6">
        <Link href="/dashboard" onClick={() => setMenuOpen(false)} className="flex items-center gap-3 text-white">
          <span aria-hidden="true" className="grid h-9 w-9 place-items-center rounded-lg border border-slate-600 text-xs font-bold">RS</span>
          <span className="text-xs font-bold leading-5 tracking-[0.15em]">RETRIEVAL<br />SHIELD</span>
        </Link>
        <button type="button" aria-expanded={menuOpen} aria-controls="primary-navigation" onClick={() => setMenuOpen(!menuOpen)} className="rounded border border-slate-600 px-3 py-2 text-sm lg:hidden">{menuOpen ? "Close menu" : "Menu"}</button>
      </div>
      <nav id="primary-navigation" aria-label="Main navigation" className={`${menuOpen ? "block" : "hidden"} space-y-7 px-3 pb-6 lg:block`}>
        {groups.map(group => <div key={group.label}><p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">{group.label}</p><ul className="space-y-1">
          {group.pages.map(([href, label]) => <li key={href}><Link href={href} aria-current={pathname === href ? "page" : undefined} onClick={() => setMenuOpen(false)} className={`block rounded-md border-l-2 px-3 py-2.5 text-sm transition ${pathname === href ? "border-sky-300 bg-slate-800 font-semibold text-white" : "border-transparent text-slate-300 hover:bg-slate-900 hover:text-white"}`}>{label}</Link></li>)}
        </ul></div>)}
      </nav>
    </aside>
    <div className="min-w-0">
      <header className="flex min-h-20 items-center justify-between gap-4 border-b border-slate-200 bg-white px-5 sm:px-8"><h1 className="text-xl font-semibold tracking-tight text-slate-900">{title}</h1><span className="hidden text-xs text-slate-500 sm:block">Retrieval Poisoning Defense</span></header>
      <main id="main-content" tabIndex={-1} className="mx-auto w-full max-w-[1440px] min-w-0 space-y-6 p-4 sm:p-6 xl:p-8">{children}</main>
    </div>
  </div>;
}
