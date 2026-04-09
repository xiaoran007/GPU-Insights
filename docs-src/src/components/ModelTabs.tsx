import type { ModelKey } from "../types/benchmark";

const TABS: { key: ModelKey | "archive"; label: string }[] = [
  { key: "vit", label: "ViT" },
  { key: "unet", label: "UNet" },
  { key: "ddpm", label: "DDPM" },
  { key: "archive", label: "Archive" },
];

interface ModelTabsProps {
  activeTab: ModelKey | "archive";
  onTabChange: (tab: ModelKey | "archive") => void;
}

export default function ModelTabs({ activeTab, onTabChange }: ModelTabsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {TABS.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => onTabChange(key)}
          className={`cursor-pointer rounded-full border px-4 py-2 text-sm font-semibold transition-colors ${
            activeTab === key
              ? "border-[var(--color-brand)] bg-[var(--color-brand)] text-[#ebfffd]"
              : "border-[var(--color-line)] bg-[var(--color-surface-soft)] text-[var(--color-text)] hover:border-blue-200 hover:bg-blue-50"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
