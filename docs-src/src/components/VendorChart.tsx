import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import type { BenchmarkEntry } from "../types/benchmark";
import { toTitleCase } from "../utils/format";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface VendorChartProps {
  entries: BenchmarkEntry[];
}

export default function VendorChart({ entries }: VendorChartProps) {
  const aggregates = new Map<
    string,
    { count: number; fp32Sum: number; fp32Count: number; fp32Peak: number }
  >();

  for (const entry of entries) {
    if (!aggregates.has(entry.vendor)) {
      aggregates.set(entry.vendor, { count: 0, fp32Sum: 0, fp32Count: 0, fp32Peak: 0 });
    }
    const agg = aggregates.get(entry.vendor)!;
    agg.count += 1;
    if (typeof entry.fp32 === "number") {
      agg.fp32Sum += entry.fp32;
      agg.fp32Count += 1;
      agg.fp32Peak = Math.max(agg.fp32Peak, entry.fp32);
    }
  }

  const sorted = [...aggregates.entries()].sort(
    ([, a], [, b]) => b.fp32Peak - a.fp32Peak,
  );

  if (sorted.length === 0) return null;

  const labels = sorted.map(([vendor]) => toTitleCase(vendor));
  const peaks = sorted.map(([, s]) => s.fp32Peak);
  const averages = sorted.map(([, s]) =>
    s.fp32Count > 0 ? Math.round(s.fp32Sum / s.fp32Count) : 0,
  );

  const data = {
    labels,
    datasets: [
      {
        label: "Peak FP32",
        data: peaks,
        backgroundColor: "rgba(15, 118, 110, 0.72)",
        borderColor: "rgba(15, 118, 110, 1)",
        borderWidth: 1,
      },
      {
        label: "Average FP32",
        data: averages,
        backgroundColor: "rgba(245, 158, 11, 0.55)",
        borderColor: "rgba(245, 158, 11, 1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <article className="rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <h2 className="m-0 font-[var(--font-display)] text-lg">Vendor Performance</h2>
      <p className="mt-1 mb-3 text-sm text-[var(--color-muted)]">
        Peak and average FP32 by vendor (filtered data)
      </p>
      <div className="relative h-80">
        <Bar
          data={data}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                beginAtZero: true,
                title: { display: true, text: "Score" },
              },
            },
            plugins: {
              legend: { position: "top" },
            },
          }}
        />
      </div>
    </article>
  );
}
