import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Retrieval Security Dashboard",
  description:
    "Review retrieval events, document trust scores, flagged sources, and answers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
