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

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface TopDevicesChartProps {
  entries: BenchmarkEntry[];
}

export default function TopDevicesChart({ entries }: TopDevicesChartProps) {
  const top = [...entries]
    .filter((e) => typeof e.fp32 === "number")
    .sort((a, b) => (b.fp32 ?? 0) - (a.fp32 ?? 0))
    .slice(0, 12);

  if (top.length === 0) return null;

  const labels = top.map((e) =>
    e.device.length > 26 ? e.device.slice(0, 26) + "…" : e.device,
  );

  const data = {
    labels,
    datasets: [
      {
        label: "FP32",
        data: top.map((e) => e.fp32),
        backgroundColor: "rgba(15, 118, 110, 0.75)",
        borderColor: "rgba(15, 118, 110, 1)",
        borderWidth: 1,
      },
      {
        label: "FP16",
        data: top.map((e) => (typeof e.fp16 === "number" ? e.fp16 : 0)),
        backgroundColor: "rgba(3, 105, 161, 0.6)",
        borderColor: "rgba(3, 105, 161, 1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <article className="rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <h2 className="m-0 font-[var(--font-display)] text-lg">Top Devices</h2>
      <p className="mt-1 mb-3 text-sm text-[var(--color-muted)]">
        FP32 and FP16 score comparison (filtered data)
      </p>
      <div className="relative h-80">
        <Bar
          data={data}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { ticks: { maxRotation: 35, minRotation: 35 } },
              y: {
                beginAtZero: true,
                title: { display: true, text: "Score" },
              },
            },
            plugins: {
              legend: { position: "top" },
              tooltip: {
                callbacks: {
                  title(items) {
                    const idx = items[0].dataIndex;
                    return top[idx].device;
                  },
                },
              },
            },
          }}
        />
      </div>
    </article>
  );
}
