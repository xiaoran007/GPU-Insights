import { useCallback, useEffect, useState } from "react";
import { api, errorMessage } from "./client";
import type { HistoryEntry, Job, Result } from "./types";

export function useWorkbench(token: string) {
  const [job, setJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const jobId = job?.id;

  const loadResult = useCallback(async (filename: string) => {
    const value = await api<Result>(`results/${encodeURIComponent(filename)}`, token);
    setResult(value);
  }, [token]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await api<Job | null>("job", token, undefined, controller.signal);
        if (controller.signal.aborted) return;
        setJob(value);
        if (value && ["running", "stopping"].includes(value.status)) {
          timer = setTimeout(poll, 800);
        } else {
          const entries = await api<HistoryEntry[]>("history", token, undefined, controller.signal);
          if (controller.signal.aborted) return;
          setHistory(entries);
          if (value?.resultFile) {
            const latest = await api<Result>(`results/${encodeURIComponent(value.resultFile)}`, token, undefined, controller.signal);
            if (!controller.signal.aborted) setResult(latest);
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) setError(errorMessage(error));
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [token, jobId]);

  return { job, setJob, history, result, setResult, loadResult, error, setError };
}
