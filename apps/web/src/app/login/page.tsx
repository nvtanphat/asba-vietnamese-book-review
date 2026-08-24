"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, AlertTriangle, LogIn, UserPlus, KeyRound, Mail, Store, User } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  const { login, register } = useAuth();
  const router = useRouter();

  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [shopName, setShopName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        await register(shopName, email, password, name);
      }
      window.location.href = "/";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin.");
    } finally {
      setLoading(false);
    }
  }

  const fillDemo = (demoEmail: string, demoPass: string) => {
    setEmail(demoEmail);
    setPassword(demoPass);
    setMode("login");
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px 16px",
        background: "var(--color-bg)",
        color: "var(--color-text)",
        fontFamily: "var(--font-body)",
      }}
    >
      <div style={{ width: "100%", maxWidth: "420px", display: "flex", flexDirection: "column", gap: "20px" }}>
        {/* Brand Header */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center", gap: "8px" }}>
          <div
            style={{
              width: "48px",
              height: "48px",
              border: "1px solid var(--color-divider)",
              background: "var(--color-surface)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--color-accent-700)",
            }}
          >
            <svg
              width="26"
              height="26"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 7v14"></path>
              <path d="M3 5a2 2 0 0 1 2-2h5a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H3V5Z"></path>
              <path d="M21 5a2 2 0 0 0-2-2h-5a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h7V5Z"></path>
            </svg>
          </div>
          <h1 className="nav-brand" style={{ fontSize: "24px", margin: 0 }}>SentenAI</h1>
          <p style={{ margin: 0, fontSize: "12px", opacity: 0.65 }}>
            Hệ thống phân tích cảm xúc đánh giá sách tiếng Việt (ABSA)
          </p>
        </div>

        {/* Blueprint Auth Card */}
        <div
          className="blueprint"
          style={{
            position: "relative",
            border: "1px solid var(--color-divider)",
            background: "var(--color-surface)",
            padding: "24px 20px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          <i className="corner tl"></i>
          <i className="corner tr"></i>
          <i className="corner bl"></i>
          <i className="corner br"></i>

          {/* Mode Switcher */}
          <div className="seg" style={{ display: "flex", width: "100%" }}>
            <span
              onClick={() => {
                setMode("login");
                setError(null);
              }}
              className={`seg-opt ${mode === "login" ? "active" : ""}`}
              style={{ flex: 1, justifyContent: "center", padding: "8px 0", fontSize: "13px" }}
            >
              <LogIn size={13} /> Đăng nhập
            </span>
            <span
              onClick={() => {
                setMode("register");
                setError(null);
              }}
              className={`seg-opt ${mode === "register" ? "active" : ""}`}
              style={{ flex: 1, justifyContent: "center", padding: "8px 0", fontSize: "13px" }}
            >
              <UserPlus size={13} /> Đăng ký shop
            </span>
          </div>

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {/* Register specific fields */}
            {mode === "register" && (
              <>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label className="card-kicker" style={{ fontSize: "10px" }}>Tên nhà sách / Cửa hàng</label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <Store size={14} style={{ position: "absolute", left: "10px", opacity: 0.4 }} />
                    <input
                      type="text"
                      required
                      value={shopName}
                      onChange={(e) => setShopName(e.target.value)}
                      placeholder="Nhà sách Minh Long"
                      className="input"
                      style={{ paddingLeft: "32px", fontSize: "13px" }}
                    />
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <label className="card-kicker" style={{ fontSize: "10px" }}>Họ tên người quản lý</label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <User size={14} style={{ position: "absolute", left: "10px", opacity: 0.4 }} />
                    <input
                      type="text"
                      required
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Ngọc Anh"
                      className="input"
                      style={{ paddingLeft: "32px", fontSize: "13px" }}
                    />
                  </div>
                </div>
              </>
            )}

            {/* Email */}
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label className="card-kicker" style={{ fontSize: "10px" }}>Địa chỉ Email</label>
              <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                <Mail size={14} style={{ position: "absolute", left: "10px", opacity: 0.4 }} />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="admin@demo.com"
                  className="input"
                  style={{ paddingLeft: "32px", fontSize: "13px" }}
                />
              </div>
            </div>

            {/* Password */}
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label className="card-kicker" style={{ fontSize: "10px" }}>Mật khẩu</label>
              <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                <KeyRound size={14} style={{ position: "absolute", left: "10px", opacity: 0.4 }} />
                <input
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="input"
                  style={{ paddingLeft: "32px", fontSize: "13px" }}
                />
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div
                className="blueprint"
                style={{
                  border: "1px solid var(--color-negative)",
                  background: "var(--color-negative-bg)",
                  padding: "10px 12px",
                  fontSize: "12px",
                  color: "var(--color-negative)",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <AlertTriangle size={14} style={{ flexShrink: 0 }} />
                <span>{error}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
              style={{ width: "100%", padding: "10px 0", marginTop: "4px", fontSize: "13px", fontWeight: 600 }}
            >
              {loading ? (
                <>
                  <Loader2 size={14} className="animate-spin" /> Đang xác thực...
                </>
              ) : mode === "login" ? (
                "Xác thực & Đăng nhập"
              ) : (
                "Hoàn tất đăng ký shop"
              )}
            </button>
          </form>
        </div>

        {/* Demo Credentials Box */}
        <div
          className="blueprint"
          style={{
            position: "relative",
            border: "1px solid var(--color-divider)",
            background: "var(--color-surface)",
            padding: "12px 16px",
            fontSize: "12px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
          }}
        >
          <i className="corner tl"></i>
          <i className="corner tr"></i>
          <i className="corner bl"></i>
          <i className="corner br"></i>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span className="card-kicker" style={{ fontSize: "10px" }}>TÀI KHOẢN TRẢI NGHIỆM NHANH (DEMO)</span>
            <span className="tag tag-outline" style={{ fontSize: "10px" }}>Sẵn sàng</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontFamily: "monospace", opacity: 0.85, fontSize: "12px" }}>
            <span>admin@demo.com / admin123</span>
            <button
              type="button"
              onClick={() => fillDemo("admin@demo.com", "admin123")}
              className="btn btn-ghost"
              style={{ padding: "2px 6px", fontSize: "11px", textDecoration: "underline", cursor: "pointer" }}
            >
              Điền nhanh
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
