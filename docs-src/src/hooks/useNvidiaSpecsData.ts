import { useEffect, useState } from "react";
import type { NvidiaCoverageData, NvidiaSpecsData } from "../types/nvidiaSpecs";

const SPECS_URL = `${import.meta.env.BASE_URL}data/nvidia-gpu-specs.json`;
const COVERAGE_URL = `${import.meta.env.BASE_URL}data/nvidia-gpu-specs-coverage.json`;

export function useNvidiaSpecsData() {
  const [data, setData] = useState<NvidiaSpecsData | null>(null);
  const [coverage, setCoverage] = useState<NvidiaCoverageData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(SPECS_URL, { cache: "no-store" }),
      fetch(COVERAGE_URL, { cache: "no-store" }),
    ])
      .then(async ([specsRes, coverageRes]) => {
        if (!specsRes.ok) throw new Error(`Specs HTTP ${specsRes.status}`);
        if (!coverageRes.ok) throw new Error(`Coverage HTTP ${coverageRes.status}`);
        const [specsPayload, coveragePayload] = await Promise.all([
          specsRes.json() as Promise<NvidiaSpecsData>,
          coverageRes.json() as Promise<NvidiaCoverageData>,
        ]);
        setData(specsPayload);
        setCoverage(coveragePayload);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { data, coverage, error, loading };
}
