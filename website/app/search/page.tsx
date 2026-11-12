import type { Metadata } from "next";
import { SearchWorkspace } from "@/components/search-workspace";

export const metadata: Metadata = { title: "Search" };

export default function Page() {
  return <SearchWorkspace />;
}
