import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/app-shell";
import { WorkspaceAccess } from "@/components/workspace-access";

export const metadata: Metadata = {
  title: { default: "Retrieval Shield", template: "%s | Retrieval Shield" },
  description:
    "Review retrieval events, document trust scores, flagged sources, and trusted passages.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body><WorkspaceAccess><AppShell>{children}</AppShell></WorkspaceAccess></body>
    </html>
  );
}
