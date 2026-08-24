import type { AbsaResult, HistorySummary, TeamMember, TikiSample } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "sentenai_token";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers as Record<string, string> ?? {}),
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
  if (!res.ok) {
    // Auto-logout on 401
    if (res.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_KEY);
      window.location.href = "/login";
      throw new Error("Session expired");
    }
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  analyze: (text: string) =>
    request<AbsaResult>("/absa/analyze", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  tikiSample: () => request<TikiSample>("/reviews/tiki-sample"),

  historySummary: (groupby: "week" | "month" = "week") =>
    request<HistorySummary>(`/reviews/history/summary?groupby=${groupby}`),

  listUsers: () => request<TeamMember[]>("/auth/users"),

  inviteUser: (payload: { name: string; email: string; password: string; role: string }) =>
    request<TeamMember>("/auth/invite", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateUser: (id: number, payload: { role?: string; is_active?: boolean }) =>
    request<TeamMember>(`/auth/users/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  getSettings: () => request<{ settings: Record<string, unknown> }>("/settings"),

  updateSettings: (settings: Record<string, unknown>) =>
    request<{ settings: Record<string, unknown> }>("/settings", {
      method: "PATCH",
      body: JSON.stringify({ settings }),
    }),
};

export { BASE_URL };
