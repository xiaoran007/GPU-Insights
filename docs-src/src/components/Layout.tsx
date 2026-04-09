import type { ReactNode } from "react";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-[#eef3f8] font-[var(--font-body)]">
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background:
            "radial-gradient(circle at 15% 5%, #d9ebff, transparent 40%), radial-gradient(circle at 90% 10%, #e7f7f0, transparent 42%)",
        }}
      />
      <main className="relative z-10 mx-auto grid w-[min(1200px,100%-2rem)] gap-4 py-6 pb-10">
        {children}
      </main>
    </div>
  );
}
