interface EmptyStateProps {
  modelName: string;
  command: string;
}

export default function EmptyState({ modelName, command }: EmptyStateProps) {
  return (
    <div className="rounded-2xl border-2 border-dashed border-[var(--color-line)] bg-[var(--color-surface-soft)] p-10 text-center">
      <div className="mx-auto mb-4 text-5xl opacity-30">📊</div>
      <h3 className="m-0 font-[var(--font-display)] text-lg text-[#334155]">
        No benchmark data yet
      </h3>
      <p className="mt-2 text-sm text-[var(--color-muted)]">
        No {modelName} benchmark results have been submitted. Run your first benchmark to
        see results here.
      </p>
      <code className="mt-3 inline-block rounded-lg bg-[#0f172a] px-4 py-2 text-sm text-[#e2f0ff]">
        {command}
      </code>
    </div>
  );
}
