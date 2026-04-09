export function formatScore(value: number | null): string {
  return typeof value === "number" ? value.toLocaleString() : "N/A";
}

export function formatBatchSize(value: number): string {
  if (value === 0) return "Auto";
  return value.toLocaleString();
}

export function toTitleCase(text: string): string {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}
