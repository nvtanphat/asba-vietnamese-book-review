"use client";

import { useState, useEffect, useCallback } from "react";
import { Plus, Users, Check, Copy, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import type { TeamMember } from "@/lib/types";

function roleLabel(role: string) {
  return role === "admin" ? "Quản lý" : "Nhân viên";
}

function statusLabel(isActive: boolean) {
  return isActive ? "Đang hoạt động" : "Vô hiệu hóa";
}

export default function SettingsPage() {
  const { shop } = useAuth();
  const [activeTab, setActiveTab] = useState<"general" | "integrations" | "team">("integrations");
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [loadingMembers, setLoadingMembers] = useState(true);
  const [copiedText, setCopiedText] = useState<string | null>(null);

  // General Settings Form state
  const [shopName, setShopName] = useState(shop?.name || "Demo Bookstore");
  const [slaWarning, setSlaWarning] = useState("4h");
  const [savingSettings, setSavingSettings] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [connectorStatus, setConnectorStatus] = useState<Record<string, boolean>>({});

  // Member Invite Form state
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteName, setInviteName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [invitePassword, setInvitePassword] = useState("");
  const [inviteRole, setInviteRole] = useState("agent");
  const [inviting, setInviting] = useState(false);

  // Sync shop name from auth context
  useEffect(() => {
    if (shop) setShopName(shop.name);
  }, [shop]);

  const loadMembers = useCallback(async () => {
    setLoadingMembers(true);
    try {
      setMembers(await api.listUsers());
    } catch {
      // non-fatal: leave list empty
    } finally {
      setLoadingMembers(false);
    }
  }, []);

  useEffect(() => {
    loadMembers();
  }, [loadMembers]);

  useEffect(() => {
    api.getSettings()
      .then(({ settings }) => {
        if (typeof settings.sla_warning === "string") setSlaWarning(settings.sla_warning);
        setConnectorStatus({
          shopee: Boolean(settings.shopee_configured),
          lazada: Boolean(settings.lazada_configured),
        });
      })
      .catch(() => {});
  }, []);

  const handleSaveGeneral = async () => {
    setSavingSettings(true);
    try {
      await api.updateSettings({ sla_warning: slaWarning });
      alert("Cập nhật cấu hình thành công!");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Cập nhật cấu hình thất bại");
    } finally {
      setSavingSettings(false);
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleConfigureConnector = async (platform: "shopee" | "lazada") => {
    try {
      await api.updateSettings({ [`${platform}_configured`]: true });
      setConnectorStatus((prev) => ({ ...prev, [platform]: true }));
      alert(`Đã lưu cấu hình kết nối ${platform === "shopee" ? "Shopee" : "Lazada"}.`);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Cập nhật cấu hình thất bại");
    }
  };

  const handleToggleMember = async (m: TeamMember) => {
    try {
      const updated = await api.updateUser(m.id, { is_active: !m.is_active });
      setMembers((prev) => prev.map((x) => (x.id === updated.id ? updated : x)));
    } catch (err) {
      alert(err instanceof Error ? err.message : "Cập nhật thành viên thất bại");
    }
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(id);
    setTimeout(() => setCopiedText(null), 2000);
  };

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteName.trim() || !inviteEmail.trim() || invitePassword.length < 6) return;

    setInviting(true);
    try {
      await api.inviteUser({
        name: inviteName,
        email: inviteEmail,
        password: invitePassword,
        role: inviteRole,
      });
      await loadMembers();
      setShowInviteModal(false);
      setInviteName("");
      setInviteEmail("");
      setInvitePassword("");
      alert(`Đã mời thành viên ${inviteEmail} tham gia thành công!`);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Mời thành viên thất bại");
    } finally {
      setInviting(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-8 bg-paper w-full max-w-[1440px] mx-auto transition-colors duration-150">
      {/* Page Title */}
      <div className="mb-8">
        <h2 className="font-display text-xl font-semibold text-ink tracking-tight">Cài đặt hệ thống</h2>
      </div>

      {/* Tabs list */}
      <div className="flex gap-6 border-b border-hairline mb-8">
        <button
          onClick={() => setActiveTab("general")}
          className={`pb-2.5 text-xs font-bold transition-all border-b-2 ${
            activeTab === "general"
              ? "text-ink border-brass"
              : "text-ink/50 hover:text-ink border-transparent"
          }`}
        >
          Cấu hình chung
        </button>
        <button
          onClick={() => setActiveTab("integrations")}
          className={`pb-2.5 text-xs font-bold transition-all border-b-2 ${
            activeTab === "integrations"
              ? "text-ink border-brass"
              : "text-ink/50 hover:text-ink border-transparent"
          }`}
        >
          Kết nối sàn TMĐT
        </button>
        <button
          onClick={() => setActiveTab("team")}
          className={`pb-2.5 text-xs font-bold transition-all border-b-2 ${
            activeTab === "team"
              ? "text-ink border-brass"
              : "text-ink/50 hover:text-ink border-transparent"
          }`}
        >
          Quản lý nhân sự
        </button>
      </div>

      {/* ── TAB CONTENT ── */}

      {/* 1. General Config Settings */}
      {activeTab === "general" && (
        <section className="max-w-xl space-y-6">
          <div className="bg-surface border border-hairline rounded-md p-6 space-y-4">
            <h3 className="font-display text-sm font-semibold text-ink">Cấu hình chung cửa hàng</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Tên Cửa hàng (Shop Name)</label>
                <input
                  type="text"
                  value={shopName}
                  onChange={(e) => setShopName(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink focus:border-brass rounded-md px-3 py-2 text-xs outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Thời hạn SLA khẩn cấp</label>
                <select
                  value={slaWarning}
                  onChange={(e) => setSlaWarning(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink rounded-md px-3 py-2 text-xs outline-none"
                >
                  <option value="2h">2 tiếng (Khẩn cấp cao)</option>
                  <option value="4h">4 tiếng (Tiêu chuẩn)</option>
                  <option value="12h">12 tiếng</option>
                  <option value="24h">24 tiếng</option>
                </select>
                <p className="text-[10px] text-ink/45 mt-1">Cảnh báo đồng hồ đếm ngược trên các bình luận tiêu cực cần xử lý nhanh.</p>
              </div>
            </div>

            <button
              onClick={handleSaveGeneral}
              disabled={savingSettings}
              className="bg-ledger hover:brightness-110 text-[var(--paper)] disabled:opacity-60 font-bold text-xs px-4 py-2 rounded-md transition-all flex items-center gap-1.5"
            >
              {savingSettings && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              Lưu thay đổi
            </button>
          </div>
        </section>
      )}

      {/* 2. Platform Integrations Settings (Connectors) — this project only supports Tiki */}
      {activeTab === "integrations" && (
        <section className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Tiki Integration Card — the only real, working integration */}
          <div className="bg-surface border border-hairline rounded-md p-6 flex flex-col relative overflow-hidden group">
            <div className="flex justify-between items-start mb-5">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-md bg-surface-sunken border border-hairline flex items-center justify-center text-brand">
                  <span className="material-symbols-outlined">shopping_bag</span>
                </div>
                <div>
                  <h3 className="font-display text-sm font-semibold text-ink">Kết nối API sàn Tiki</h3>
                  <p className="text-[10px] text-ink/50 font-semibold uppercase tracking-wider">Nền tảng TMĐT duy nhất được hỗ trợ</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-ledger/10 border border-ledger/25 text-[10px] text-brand flex items-center gap-1 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-ledger"></span>
                Đã kết nối
              </span>
            </div>

            <div className="space-y-4 flex-1">
              <p className="text-xs text-ink/65 leading-relaxed">
                Hệ thống lấy review trực tiếp qua endpoint JSON công khai của Tiki
                (<code className="font-mono text-[10px]">tiki.vn/api/v2/reviews</code>), không cần
                OAuth hay API key. Dùng nút <span className="font-semibold">&quot;Lấy review từ Tiki&quot;</span> ở
                trang Phân tích ABSA để nạp review thật vào bộ phân tích.
              </p>
              <div>
                <label className="block text-[10px] font-bold text-ink/45 uppercase tracking-wider mb-1">Endpoint đang dùng</label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    readOnly
                    value="GET /reviews/tiki-sample"
                    className="flex-1 bg-surface-sunken border border-hairline text-ink/70 rounded-md px-3 py-1.5 font-mono text-xs outline-none"
                  />
                  <button
                    onClick={() => handleCopy("/reviews/tiki-sample", "tiki")}
                    className="px-2.5 py-1.5 bg-surface border border-hairline hover:bg-surface-sunken text-ink/60 hover:text-ink rounded-md transition-colors"
                  >
                    {copiedText === "tiki" ? <Check className="h-4 w-4 text-brand" /> : <Copy className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-5 border-t border-hairline text-[10px]">
              <span className="text-ink/45 font-medium">Gọi API trực tiếp, không cache</span>
            </div>
          </div>
        </section>
      )}

      {/* 3. Team Management Table */}
      {activeTab === "team" && (
        <section className="space-y-6">
          <div className="flex justify-between items-center">
            <h3 className="font-display text-sm font-semibold text-ink flex items-center gap-1.5">
              <Users className="h-4 w-4 text-brand" />
              Thành viên cửa hàng
            </h3>
            <button
              onClick={() => setShowInviteModal(true)}
              className="bg-ledger hover:brightness-110 text-[var(--paper)] font-bold text-xs px-3.5 py-1.5 rounded-md flex items-center gap-1 transition-all"
            >
              <Plus className="h-4 w-4" />
              Mời nhân viên
            </button>
          </div>

          <div className="bg-surface border border-hairline rounded-md overflow-hidden">
            {loadingMembers ? (
              <div className="flex items-center justify-center gap-2 py-10 text-ink/50 text-xs">
                <Loader2 className="h-4 w-4 animate-spin text-brass" /> Đang tải danh sách thành viên...
              </div>
            ) : (
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-hairline bg-surface-sunken text-ink/55 font-bold uppercase tracking-wider">
                  <th className="p-4">Họ tên</th>
                  <th className="p-4">Địa chỉ Email</th>
                  <th className="p-4">Vai trò quản trị</th>
                  <th className="p-4">Trạng thái tài khoản</th>
                  <th className="p-4"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-hairline">
                {members.map((m) => (
                  <tr key={m.id} className="hover:bg-surface-sunken/60 transition-colors text-ink">
                    <td className="p-4 font-semibold">{m.name}</td>
                    <td className="p-4 font-mono text-ink/55">{m.email}</td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${m.role === "admin" ? "bg-brass/10 text-brass border border-brass/25" : "bg-surface-sunken text-ink/60 border border-hairline"}`}>
                        {roleLabel(m.role)}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`font-semibold ${m.is_active ? "text-brand" : "text-brass"}`}>
                        {statusLabel(m.is_active)}
                      </span>
                    </td>
                    <td className="p-4 text-right">
                      <button
                        onClick={() => handleToggleMember(m)}
                        className="text-[10px] font-semibold text-slate-500 hover:text-indigo-600 dark:text-slate-400 dark:hover:text-indigo-400"
                      >
                        {m.is_active ? "Vô hiệu hóa" : "Kích hoạt lại"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            )}
          </div>
        </section>
      )}

      {/* ── MEMBER INVITE MODAL ── */}
      {showInviteModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <form onSubmit={handleInvite} className="bg-surface border border-hairline rounded-md max-w-md w-full overflow-hidden shadow-2xl">
            <div className="px-6 py-4 bg-surface-sunken border-b border-hairline flex justify-between items-center text-ink">
              <h3 className="text-sm font-bold uppercase tracking-wider">Mời thành viên tham gia</h3>
              <button type="button" onClick={() => setShowInviteModal(false)} className="text-ink/45 hover:text-ink">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Họ tên nhân viên</label>
                <input
                  type="text"
                  required
                  placeholder="Ví dụ: Nguyễn Văn B"
                  value={inviteName}
                  onChange={(e) => setInviteName(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink rounded-md px-3 py-2 text-xs outline-none focus:border-brass"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Địa chỉ Email</label>
                <input
                  type="email"
                  required
                  placeholder="cskh@yourshop.com"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink rounded-md px-3 py-2 text-xs outline-none focus:border-brass"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Mật khẩu tạm thời</label>
                <input
                  type="text"
                  required
                  minLength={6}
                  placeholder="Tối thiểu 6 ký tự"
                  value={invitePassword}
                  onChange={(e) => setInvitePassword(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink rounded-md px-3 py-2 text-xs outline-none focus:border-brass"
                />
              </div>

              <div>
                <label className="block text-xs font-bold text-ink/50 uppercase tracking-wider mb-1.5">Phân quyền quản trị</label>
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  className="w-full bg-surface-sunken border border-hairline text-ink rounded-md px-3 py-2 text-xs outline-none"
                >
                  <option value="agent">Nhân viên CSKH (Agent)</option>
                  <option value="admin">Quản lý cửa hàng (Admin)</option>
                </select>
              </div>
            </div>

            <div className="px-6 py-4 bg-surface-sunken border-t border-hairline flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setShowInviteModal(false)}
                className="px-4 py-2 bg-transparent border border-hairline text-ink/65 hover:bg-surface rounded-md text-xs font-semibold"
              >
                Hủy
              </button>
              <button
                type="submit"
                disabled={inviting}
                className="px-5 py-2 bg-ledger hover:brightness-110 text-[var(--paper)] disabled:opacity-60 rounded-md text-xs font-bold transition-all"
              >
                {inviting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                Gửi lời mời
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
