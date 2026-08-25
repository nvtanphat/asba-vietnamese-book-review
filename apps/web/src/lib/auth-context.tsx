"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

// ── Types ────────────────────────────────────────────────────────────────

export interface AuthUser {
  id: number;
  email: string;
  name: string;
  role: "admin" | "agent";
  shop_id: number;
}

export interface AuthShop {
  id: number;
  name: string;
  slug: string;
  plan: string;
}

interface AuthState {
  user: AuthUser | null;
  shop: AuthShop | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (shopName: string, email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

// ── Helpers ──────────────────────────────────────────────────────────────

const TOKEN_KEY = "sentenai_token";
const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(body || `API ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Provider ─────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [shop, setShop] = useState<AuthShop | null>(null);
  const [loading, setLoading] = useState(true);

  // Load user profile from a real token; any failure (expired/invalid token) clears
  // the session so the route guard sends the user back to a real login, never a fake one.
  const loadMe = useCallback(async (t: string) => {
    try {
      const me = await apiRequest<{ user: AuthUser; shop: AuthShop }>("/auth/me", {
        headers: { Authorization: `Bearer ${t}` },
      });
      setUser(me.user);
      setShop(me.shop);
      setToken(t);
    } catch {
      localStorage.removeItem(TOKEN_KEY);
      setToken(null);
      setUser(null);
      setShop(null);
    }
  }, []);

  // On mount, only restore a session if a real stored token still validates against the
  // API. No token (or an invalid one) leaves user/shop null so AppShell's route guard
  // redirects to /login — there is no "default to a demo session" bypass.
  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (stored) {
      loadMe(stored).finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [loadMe]);

  const login = useCallback(
    async (email: string, password: string) => {
      const data = await apiRequest<{ access_token: string }>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      localStorage.setItem(TOKEN_KEY, data.access_token);
      await loadMe(data.access_token);
    },
    [loadMe],
  );

  const register = useCallback(
    async (shopName: string, email: string, password: string, name: string) => {
      const data = await apiRequest<{ access_token: string }>("/auth/register", {
        method: "POST",
        body: JSON.stringify({ shop_name: shopName, email, password, name }),
      });
      localStorage.setItem(TOKEN_KEY, data.access_token);
      await loadMe(data.access_token);
    },
    [loadMe],
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    setShop(null);
  }, []);

  const value = useMemo(
    () => ({ user, shop, token, loading, login, register, logout }),
    [user, shop, token, loading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// ── Hook ─────────────────────────────────────────────────────────────────

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}

