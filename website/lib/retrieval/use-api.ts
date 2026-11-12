"use client";

import { useCallback, useEffect, useState } from "react";

export function useApi<T>(load: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);
  const refresh = useCallback(() => setVersion(value => value + 1), []);
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    load().then(value => { if (active) setData(value); }).catch((failure: unknown) => {
      if (active) { setData(null); setError(failure instanceof Error ? failure.message : "Unable to load data."); }
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [load, version]);
  return { data, loading, error, refresh };
}
