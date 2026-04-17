interface HeroProps {
  lastUpdated: string;
  eyebrow?: string;
  title?: string;
  description?: string;
  lastUpdatedLabel?: string;
}

export default function Hero({
  lastUpdated,
  eyebrow = "GitHub Pages Dashboard",
  title = "GPU Insights",
  description = "Multi-model training benchmark results across GPUs and NPUs.",
  lastUpdatedLabel = "Last updated",
}: HeroProps) {
  return (
    <header
      className="relative overflow-hidden rounded-2xl border-0 p-5 text-[#f8fdff]"
      style={{
        background: "linear-gradient(132deg, #0f172a 0%, #0f766e 58%, #0284c7 100%)",
      }}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at 85% 25%, rgba(255,255,255,0.2), transparent 45%)",
        }}
      />
      <div className="relative z-10">
        <p className="m-0 text-xs uppercase tracking-widest opacity-85">
          {eyebrow}
        </p>
        <h1 className="my-1.5 font-[var(--font-display)] text-[clamp(2rem,6vw,3rem)] leading-tight tracking-tight">
          {title}
        </h1>
        <p className="m-0 text-[rgba(245,252,255,0.9)]">
          {description}
        </p>
        <p className="mt-1.5 m-0 text-sm text-[rgba(245,252,255,0.9)]">
          {lastUpdatedLabel}: {lastUpdated}
        </p>
      </div>
    </header>
  );
}
