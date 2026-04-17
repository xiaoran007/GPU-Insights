type ViewKey = "benchmarks" | "nvidia-specs";

const PAGES: { key: ViewKey; label: string }[] = [
  { key: "benchmarks", label: "Benchmarks" },
  { key: "nvidia-specs", label: "NVIDIA Specs" },
];

interface PageTabsProps {
  activeView: ViewKey;
  onViewChange: (view: ViewKey) => void;
}

export default function PageTabs({ activeView, onViewChange }: PageTabsProps) {
  return (
    <nav className="flex flex-wrap gap-2">
      {PAGES.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => onViewChange(key)}
          className={`cursor-pointer rounded-full border px-4 py-2 text-sm font-semibold transition-colors ${
            activeView === key
              ? "border-[var(--color-brand)] bg-[var(--color-brand)] text-[#ebfffd]"
              : "border-[var(--color-line)] bg-white text-[var(--color-text)] hover:border-sky-200 hover:bg-sky-50"
          }`}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}
