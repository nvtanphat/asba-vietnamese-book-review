import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-ledger text-[var(--paper)] shadow-[0_1px_0_rgba(0,0,0,0.15)] hover:brightness-110 hover:-translate-y-px active:translate-y-0 disabled:opacity-50 disabled:hover:brightness-100",
  secondary:
    "bg-surface text-ink ring-1 ring-hairline hover:bg-surface-sunken",
  ghost: "bg-transparent hover:bg-surface-sunken text-ink/80",
};

export function Button({
  variant = "primary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-semibold",
        "transition-all duration-150 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brass/50 focus:ring-offset-2 focus:ring-offset-[var(--paper)]",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}
