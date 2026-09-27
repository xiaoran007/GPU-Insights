import { useEffect, useState } from "react";
import type { LlmApiData } from "../types/llmApi";

const DATA_URL = `${import.meta.env.BASE_URL}data/llm-api-data.json`;

export function useLlmApiData() {
  const [data, setData] = useState<LlmApiData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(DATA_URL, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((payload: LlmApiData) => {
        setData(payload);
        setLoading(false);
      })
      .catch((reason: Error) => {
        setError(reason.message);
        setLoading(false);
      });
  }, []);

  return { data, error, loading };
}
