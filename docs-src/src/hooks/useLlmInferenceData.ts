import { useEffect, useState } from "react";
import type { LlmInferenceData } from "../types/llmInference";

const DATA_URL = `${import.meta.env.BASE_URL}data/llm-inference-data.json`;

export function useLlmInferenceData() {
  const [data, setData] = useState<LlmInferenceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(DATA_URL, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((payload: LlmInferenceData) => {
        setData(payload);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { data, error, loading };
}
