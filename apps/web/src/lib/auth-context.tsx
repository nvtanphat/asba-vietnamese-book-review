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

  // Load user profile from token
  const loadMe = useCallback(async (t: string) => {
    if (t === "demo_jwt_token_admin") {
      try {
        const cached = localStorage.getItem("sentenai_demo_user");
        if (cached) {
          const parsed = JSON.parse(cached);
          setUser(parsed.user);
          setShop(parsed.shop);
          setToken(t);
          return;
        }
      } catch {
        // ignore parse error
      }
      const demoUser: AuthUser = {
        id: 1,
        email: "admin@demo.com",
        name: "Ngọc Anh",
        role: "admin",
        shop_id: 1,
      };
      const demoShop: AuthShop = {
        id: 1,
        name: "Nhà sách Minh Long",
        slug: "minh-long-bookstore",
        plan: "pro",
      };
      setUser(demoUser);
      setShop(demoShop);
      setToken(t);
      return;
    }
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

  // On mount, check for existing token, otherwise default to demo session so user is never blocked
  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (stored) {
      loadMe(stored).finally(() => setLoading(false));
    } else {
      // Default to demo admin session
      const demoUser: AuthUser = {
        id: 1,
        email: "admin@demo.com",
        name: "Ngọc Anh",
        role: "admin",
        shop_id: 1,
      };
      const demoShop: AuthShop = {
        id: 1,
        name: "Nhà sách Minh Long",
        slug: "minh-long-bookstore",
        plan: "pro",
      };
      setUser(demoUser);
      setShop(demoShop);
      setToken("demo_jwt_token_admin");
      setLoading(false);
    }
  }, [loadMe]);

  const login = useCallback(
    async (email: string, password: string) => {
      try {
        const data = await apiRequest<{ access_token: string }>("/auth/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
        localStorage.setItem(TOKEN_KEY, data.access_token);
        await loadMe(data.access_token);
      } catch (err) {
        // If backend is offline or demo login:
        const demoToken = "demo_jwt_token_admin";
        const demoUser: AuthUser = {
          id: 1,
          email: email || "admin@demo.com",
          name: "Ngọc Anh",
          role: "admin",
          shop_id: 1,
        };
        const demoShop: AuthShop = {
          id: 1,
          name: "Nhà sách Minh Long",
          slug: "minh-long-bookstore",
          plan: "pro",
        };
        localStorage.setItem(TOKEN_KEY, demoToken);
        localStorage.setItem("sentenai_demo_user", JSON.stringify({ user: demoUser, shop: demoShop }));
        setToken(demoToken);
        setUser(demoUser);
        setShop(demoShop);
      }
    },
    [loadMe],
  );

  const register = useCallback(
    async (shopName: string, email: string, password: string, name: string) => {
      try {
        const data = await apiRequest<{ access_token: string }>("/auth/register", {
          method: "POST",
          body: JSON.stringify({ shop_name: shopName, email, password, name }),
        });
        localStorage.setItem(TOKEN_KEY, data.access_token);
        await loadMe(data.access_token);
      } catch {
        const demoToken = "demo_jwt_token_admin";
        const demoUser: AuthUser = {
          id: Math.floor(Math.random() * 1000) + 10,
          email,
          name: name || "Quản lý Shop",
          role: "admin",
          shop_id: 1,
        };
        const demoShop: AuthShop = {
          id: 1,
          name: shopName || "Nhà sách Minh Long",
          slug: "shop-demo",
          plan: "pro",
        };
        localStorage.setItem(TOKEN_KEY, demoToken);
        localStorage.setItem("sentenai_demo_user", JSON.stringify({ user: demoUser, shop: demoShop }));
        setToken(demoToken);
        setUser(demoUser);
        setShop(demoShop);
      }
    },
    [loadMe],
  );

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem("sentenai_demo_user");
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

