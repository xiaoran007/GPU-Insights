import type { ModelInfo } from "../types/benchmark";

interface ModelProfileProps {
  model: ModelInfo;
}

export default function ModelProfile({ model }: ModelProfileProps) {
  return (
    <section className="rounded-2xl border border-[var(--color-line)] bg-white p-5 shadow-[0_14px_36px_rgba(15,23,42,0.08)]">
      <h2 className="m-0 font-[var(--font-display)] text-lg">Benchmark Profile</h2>
      <p className="mt-2 mb-3 text-sm text-[var(--color-muted)]">
        Scores represent completed training throughput. Higher values indicate stronger
        compute performance.
      </p>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-3">
        <ProfileItem label="Model" value={model.displayName} />
        <ProfileItem label="Parameters" value={model.parameters} />
        <ProfileItem label="Input Size" value={model.inputSize} />
        <ProfileItem label="Task" value={model.task} />
        <div className="col-span-full">
          <ProfileItem label="Default Command" value={model.defaultCommand} isCode />
        </div>
      </div>
    </section>
  );
}

function ProfileItem({
  label,
  value,
  isCode = false,
}: {
  label: string;
  value: string;
  isCode?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl border border-[var(--color-line)] bg-[var(--color-surface-soft)] px-3.5 py-2.5">
      <span className="text-xs text-[var(--color-muted)]">{label}</span>
      {isCode ? (
        <code className="inline-block w-fit max-w-full rounded-lg bg-[#0f172a] px-2.5 py-1.5 text-sm text-[#e2f0ff] break-all">
          {value}
        </code>
      ) : (
        <span className="font-semibold">{value}</span>
      )}
    </div>
  );
}
