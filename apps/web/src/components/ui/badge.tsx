import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type Tone = "slate" | "red" | "amber" | "green" | "indigo" | "sky";

const TONES: Record<Tone, string> = {
  slate: "bg-surface-sunken text-ink/70 ring-hairline",
  red: "bg-[var(--wax-seal)]/10 text-[var(--wax-seal)] ring-[var(--wax-seal)]/30",
  amber: "bg-[var(--brass)]/10 text-[var(--brass)] ring-[var(--brass)]/30",
  green: "bg-[var(--ledger)]/10 text-[var(--ledger)] ring-[var(--ledger)]/25 dark:text-[var(--ledger-strong)]",
  indigo: "bg-[var(--ledger)]/10 text-[var(--ledger)] ring-[var(--ledger)]/25 dark:text-[var(--ledger-strong)]",
  sky: "bg-surface-sunken text-ink/70 ring-hairline",
};

export function Badge({
  tone = "slate",
  className,
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold tracking-wide ring-1 ring-inset transition-colors",
        TONES[tone],
        className,
      )}
      {...props}
    />
  );
}
