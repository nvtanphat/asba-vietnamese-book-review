// Client-side session history for the ABSA analyzer. There is no backend table for
// analyzed reviews anymore (each /absa/analyze call is stateless) — this persists the
// running session log to localStorage so the dashboard's stats/report page can read the
// same data the homepage accumulates, across a client-side route change or a reload.
import type { AbsaResult } from "./types";

const STORAGE_KEY = "sentenai_session_history";

export interface HistoryEntry {
  result: AbsaResult;
  timestamp: string; // ISO
}

export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function appendHistory(result: AbsaResult): HistoryEntry[] {
  const next = [...loadHistory(), { result, timestamp: new Date().toISOString() }];
  if (typeof window !== "undefined") {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }
  return next;
}

export function clearHistory(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}
