"use client";

import { useEffect, useMemo, useState } from "react";
import {
  RotateCcw,
  Trash2,
  Cloud,
  Monitor,
  Search,
  BookOpen,
  Layers,
  DollarSign,
  Package,
  Truck,
  Headphones,
} from "lucide-react";
import { api } from "@/lib/api";
import { clearHistory, loadHistory, type HistoryEntry } from "@/lib/session-history";
import type { HistorySummaryBucket } from "@/lib/types";

const ASPECT_LABELS: Record<string, string> = {
  as_content: "Nội dung sách",
  as_physical: "Hình thức vật lý",
  as_price: "Giá cả",
  as_packaging: "Đóng gói",
  as_delivery: "Giao hàng",
  as_service: "Dịch vụ / Tư vấn",
};

const ASPECT_CONFIG = [
  { key: "as_content", name: "Nội dung sách", icon: BookOpen },
  { key: "as_physical", name: "Hình thức vật lý", icon: Layers },
  { key: "as_price", name: "Giá cả", icon: DollarSign },
  { key: "as_packaging", name: "Đóng gói", icon: Package },
  { key: "as_delivery", name: "Giao hàng", icon: Truck },
  { key: "as_service", name: "Dịch vụ / Tư vấn", icon: Headphones },
];

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString("vi-VN", {
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

export default function StatsPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [groupBy, setGroupBy] = useState<"week" | "month">("week");
  const [buckets, setBuckets] = useState<HistorySummaryBucket[] | null>(null);
  const [loadingBuckets, setLoadingBuckets] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState<string>("all");

  // Load session history on mount
  useEffect(() => {
    setEntries(loadHistory());
  }, []);

  // Fetch long-term server-side summary
  useEffect(() => {
    setLoadingBuckets(true);
    api
      .historySummary(groupBy)
      .then((res) => {
        setBuckets(res.buckets);
      })
      .catch(() => {
        // Fallback default sample data if API server is in local offline mode
        setBuckets(null);
      })
      .finally(() => setLoadingBuckets(false));
  }, [groupBy]);

  const refreshHistory = () => {
    setEntries(loadHistory());
  };

  const handleClearHistory = () => {
    if (confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử phiên làm việc hiện tại?")) {
      clearHistory();
      setEntries([]);
    }
  };

  // ── Long-term trend data computation ──────────────────────────────────
  const trendBars = useMemo(() => {
    if (buckets && buckets.length > 0) {
      const maxTotal = Math.max(...buckets.map((b) => b.total), 1);
      return buckets.map((b) => {
        const hPos = Math.round((b.positive / maxTotal) * 110);
        const hNeu = Math.round((b.neutral / maxTotal) * 110);
        const hNeg = Math.round((b.negative / maxTotal) * 110);
        const posPct = Math.round((b.positive / b.total) * 100);
        const neuPct = Math.round((b.neutral / b.total) * 100);
        const negPct = Math.max(0, 100 - posPct - neuPct);
        const label = groupBy === "week" ? b.period.replace(/^\d{4}-/, "") : b.period;
        return { label, hPos, hNeu, hNeg, posPct, neuPct, negPct };
      });
    }

    // No fabricated chart when the server has no history yet — the caller renders an
    // explicit empty state for an empty array instead.
    return [];
  }, [buckets, groupBy]);

  // ── Session Metrics Computation ────────────────────────────────────────
  const sessionOverview = useMemo(() => {
    const total = entries.length;
    if (total === 0) {
      return {
        total: 0,
        posPct: 0,
        neuPct: 0,
        negPct: 0,
        topAspect: "",
        avgConfidence: 0,
        ranking: ASPECT_CONFIG.map((c) => ({ name: c.name, count: 0, barPct: 0 })),
        sparklinePoints: "",
        startPct: 0,
        endPct: 0,
        deltaAbs: 0,
        donut: { posDash: "0 314.16", posOffset: "0", neuDash: "0 314.16", neuOffset: "0", negDash: "0 314.16", negOffset: "0" },
      };
    }

    let pos = 0,
      neu = 0,
      neg = 0;
    let confSum = 0;
    const aspectNegCounts: Record<string, number> = {};
    ASPECT_CONFIG.forEach((c) => (aspectNegCounts[c.key] = 0));

    const sparklineData: number[] = [];

    entries.forEach((e, idx) => {
      const r = e.result;
      if (r.overall === "positive") pos++;
      else if (r.overall === "neutral") neu++;
      else if (r.overall === "negative") neg++;

      const sIdx = r.overall === "negative" ? 0 : r.overall === "neutral" ? 1 : 2;
      confSum += (r.overall_probs?.[sIdx] ?? 0.85) * 100;

      r.aspects?.forEach((a) => {
        if (a.presence > 0.45 && a.sentiment === "negative") {
          aspectNegCounts[a.aspect] = (aspectNegCounts[a.aspect] || 0) + 1;
        }
      });

      const runningNegRate = (neg / (idx + 1)) * 100;
      sparklineData.push(runningNegRate);
    });

    const posPct = Math.round((pos / total) * 100);
    const neuPct = Math.round((neu / total) * 100);
    const negPct = Math.max(0, 100 - posPct - neuPct);
    const avgConfidence = Math.round(confSum / total);

    const ranking = ASPECT_CONFIG.map((c) => ({
      name: c.name,
      count: aspectNegCounts[c.key] || 0,
      barPct: 0,
    })).sort((a, b) => b.count - a.count);

    const maxCount = Math.max(1, ranking[0].count);
    ranking.forEach((r) => (r.barPct = Math.round((r.count / maxCount) * 100)));

    const topAspect = ranking[0].count > 0 ? ranking[0].name : "Không có";

    // Donut math (circumference = 2 * PI * 50 = 314.16)
    const C = 314.16;
    const posLen = (posPct / 100) * C;
    const neuLen = (neuPct / 100) * C;
    const negLen = (negPct / 100) * C;

    const donut = {
      posDash: `${posLen.toFixed(1)} ${(C - posLen).toFixed(1)}`,
      posOffset: "0",
      neuDash: `${neuLen.toFixed(1)} ${(C - neuLen).toFixed(1)}`,
      neuOffset: `${(-posLen).toFixed(1)}`,
      negDash: `${negLen.toFixed(1)} ${(C - negLen).toFixed(1)}`,
      negOffset: `${(-(posLen + neuLen)).toFixed(1)}`,
    };

    // Sparkline math
    const startPct = Math.round(sparklineData[0] || 0);
    const endPct = Math.round(sparklineData[sparklineData.length - 1] || 0);
    const deltaAbs = Math.abs(endPct - startPct);

    let sparklinePoints = "0,50 300,50";
    if (sparklineData.length > 1) {
      const maxVal = Math.max(...sparklineData, 50);
      sparklinePoints = sparklineData
        .map((val, i) => {
          const x = (i / (sparklineData.length - 1)) * 300;
          const y = 100 - (val / maxVal) * 80 - 10;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    }

    return {
      total,
      posPct,
      neuPct,
      negPct,
      topAspect,
      avgConfidence,
      ranking,
      sparklinePoints,
      startPct,
      endPct,
      deltaAbs,
      donut,
    };
  }, [entries]);

  // ── Filtered Session Log Table ─────────────────────────────────────────
  const filteredLog = useMemo(() => {
    let list = entries.map((e) => {
      const r = e.result;
      const negAspects = r.aspects
        ?.filter((a) => a.presence > 0.45 && a.sentiment === "negative")
        .map((a) => ASPECT_LABELS[a.aspect] || a.aspect) || [];

      let sentimentLabel = "Tích cực";
      let sentimentColor = "var(--color-accent-700)";
      let sentimentBg = "var(--color-accent-100)";

      if (r.overall === "negative") {
        sentimentLabel = "Tiêu cực";
        sentimentColor = "var(--color-negative)";
        sentimentBg = "var(--color-negative-bg)";
      } else if (r.overall === "neutral") {
        sentimentLabel = "Trung lập";
        sentimentColor = "var(--color-neutral-700)";
        sentimentBg = "var(--color-neutral-200)";
      }

      return {
        time: formatTime(e.timestamp),
        sentiment: r.overall,
        sentimentLabel,
        sentimentColor,
        sentimentBg,
        negAspects,
        snippet: r.text || "Đánh giá sách không có nội dung văn bản",
      };
    });

    // If no entries in session yet, use default mock rows
    if (list.length === 0) {
      list = [
        {
          time: "14:32",
          sentiment: "positive",
          sentimentLabel: "Tích cực",
          sentimentColor: "var(--color-accent-700)",
          sentimentBg: "var(--color-accent-100)",
          negAspects: [],
          snippet: "Sách đóng gói chắc chắn, giao nhanh hơn dự kiến, nội dung rất cuốn hút…",
        },
        {
          time: "14:26",
          sentiment: "negative",
          sentimentLabel: "Tiêu cực",
          sentimentColor: "var(--color-negative)",
          sentimentBg: "var(--color-negative-bg)",
          negAspects: ["Giao hàng", "Giá cả"],
          snippet: "Chờ giao hàng gần 2 tuần, giá lại đắt hơn nhà sách khác, khá thất vọng…",
        },
        {
          time: "14:19",
          sentiment: "neutral",
          sentimentLabel: "Trung lập",
          sentimentColor: "var(--color-neutral-700)",
          sentimentBg: "var(--color-neutral-200)",
          negAspects: ["Hình thức vật lý"],
          snippet: "Sách ổn, bìa hơi mỏng, nội dung chưa có gì đặc biệt…",
        },
        {
          time: "14:11",
          sentiment: "positive",
          sentimentLabel: "Tích cực",
          sentimentColor: "var(--color-accent-700)",
          sentimentBg: "var(--color-accent-100)",
          negAspects: [],
          snippet: "Dịch giả dịch rất mượt, giấy đẹp, đóng gói kỹ, sẽ ủng hộ tiếp…",
        },
        {
          time: "14:03",
          sentiment: "negative",
          sentimentLabel: "Tiêu cực",
          sentimentColor: "var(--color-negative)",
          sentimentBg: "var(--color-negative-bg)",
          negAspects: ["Giao hàng"],
          snippet: "Shipper giao trễ 3 ngày so với cam kết, thái độ không thân thiện…",
        },
        {
          time: "13:58",
          sentiment: "positive",
          sentimentLabel: "Tích cực",
          sentimentColor: "var(--color-accent-700)",
          sentimentBg: "var(--color-accent-100)",
          negAspects: [],
          snippet: "Giá hợp lý, nội dung sách bổ ích, đóng gói an toàn…",
        },
        {
          time: "13:51",
          sentiment: "negative",
          sentimentLabel: "Tiêu cực",
          sentimentColor: "var(--color-negative)",
          sentimentBg: "var(--color-negative-bg)",
          negAspects: ["Đóng gói", "Giao hàng"],
          snippet: "Sách bị móp góc do đóng gói sơ sài, giao hàng cũng chậm…",
        },
        {
          time: "13:44",
          sentiment: "neutral",
          sentimentLabel: "Trung lập",
          sentimentColor: "var(--color-neutral-700)",
          sentimentBg: "var(--color-neutral-200)",
          negAspects: ["Dịch vụ / Tư vấn"],
          snippet: "Liên hệ CSKH hỏi đổi trả nhưng phản hồi khá chậm, sách thì bình thường…",
        },
      ];
    }

    if (sentimentFilter !== "all") {
      list = list.filter((r) => r.sentiment === sentimentFilter);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (r) =>
          r.snippet.toLowerCase().includes(q) ||
          r.negAspects.some((na) => na.toLowerCase().includes(q))
      );
    }

    return list;
  }, [entries, sentimentFilter, searchQuery]);

  return (
    <div className="flex flex-col gap-6">
      {/* ── SCREEN TITLE LABEL ─────────────────────────────────────── */}
      <div className="font-heading text-xs uppercase tracking-wider text-[var(--color-accent-700)]">
        Màn hình · Thống kê & Báo cáo
      </div>

      {/* ── SECTION 1: LONG-TERM TREND (Server-side) ───────────────── */}
      <div className="blueprint p-5 flex flex-col gap-4">
        <i className="corner tl"></i>
        <i className="corner tr"></i>
        <i className="corner bl"></i>
        <i className="corner br"></i>

        <div className="flex items-center gap-3 flex-wrap">
          <div>
            <div className="card-kicker">DỮ LIỆU MÁY CHỦ</div>
            <div className="card-title text-xl">Xu hướng dài hạn</div>
          </div>

          <span className="tag tag-outline flex items-center gap-1.5 ml-2">
            <Cloud size={12} />
            Không mất khi tải lại trang
          </span>

          <div className="seg ml-auto">
            <span
              onClick={() => setGroupBy("week")}
              className={`seg-opt ${groupBy === "week" ? "active" : ""}`}
            >
              Tuần
            </span>
            <span
              onClick={() => setGroupBy("month")}
              className={`seg-opt ${groupBy === "month" ? "active" : ""}`}
            >
              Tháng
            </span>
          </div>
        </div>

        {/* Stacked Bars Chart */}
        <div className="flex items-end gap-4 h-44 mt-2 px-1">
          {trendBars.length === 0 && (
            <span className="text-sm opacity-55 self-center">
              {loadingBuckets ? "Đang tải..." : "Chưa có dữ liệu — hãy phân tích ít nhất một review để bắt đầu tích lũy xu hướng."}
            </span>
          )}
          {trendBars.map((w, idx) => (
            <div key={idx} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end">
              <div className="flex flex-col w-full max-w-[36px]">
                <div
                  style={{ height: `${w.hNeg}px` }}
                  className="bg-[var(--color-negative)] transition-all duration-300"
                ></div>
                <div
                  style={{ height: `${w.hNeu}px` }}
                  className="bg-[var(--color-neutral-500)] transition-all duration-300"
                ></div>
                <div
                  style={{ height: `${w.hPos}px` }}
                  className="bg-[var(--color-accent-700)] transition-all duration-300"
                ></div>
              </div>
              <span className="text-[11px] opacity-60 font-medium">{w.label}</span>
            </div>
          ))}
        </div>

        {/* Bottom Ratio Bars */}
        <div className="flex items-center gap-4 px-1">
          {trendBars.map((w, idx) => (
            <div
              key={idx}
              className="flex-1 max-w-[36px] mx-auto h-2 flex overflow-hidden border border-[var(--color-divider)]"
            >
              <div style={{ width: `${w.posPct}%` }} className="bg-[var(--color-accent-700)]"></div>
              <div style={{ width: `${w.neuPct}%` }} className="bg-[var(--color-neutral-500)]"></div>
              <div style={{ width: `${w.negPct}%` }} className="bg-[var(--color-negative)]"></div>
            </div>
          ))}
        </div>

        {/* Caption & Legend */}
        <div className="flex justify-between items-center text-xs opacity-75 mt-1 flex-wrap gap-2">
          <span>
            {groupBy === "month" ? "6 tháng gần nhất" : "8 tuần gần nhất"} · trên: số lượng theo cảm xúc · dưới: tỷ lệ %
          </span>
          <div className="flex gap-3.5 font-medium">
            <span className="text-[var(--color-accent-700)]">● Tích cực</span>
            <span className="text-[var(--color-neutral-600)]">● Trung lập</span>
            <span className="text-[var(--color-negative)]">● Tiêu cực</span>
          </div>
        </div>
      </div>

      {/* ── SECTION 2: SESSION STATS (Browser Session) ─────────────── */}
      <div className="flex flex-col gap-5">
        {/* Section Header */}
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="card-title text-2xl">Thống kê phiên làm việc hiện tại</h2>
          <span className="tag tag-outline flex items-center gap-1.5">
            <Monitor size={12} />
            Chỉ lưu trên trình duyệt này
          </span>
          <span className="text-xs opacity-55 hidden md:inline">
            Mất khi đóng trình duyệt hoặc bấm &ldquo;Xóa lịch sử&rdquo;
          </span>

          <div className="ml-auto flex items-center gap-2">
            <button onClick={refreshHistory} className="btn btn-secondary text-xs">
              <RotateCcw size={13} /> Làm mới
            </button>
            <button
              onClick={handleClearHistory}
              className="btn btn-secondary text-xs text-[var(--color-negative)] border-[var(--color-negative)]/30"
            >
              <Trash2 size={13} /> Xóa lịch sử phiên
            </button>
          </div>
        </div>

        {/* 4 Metric Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="blueprint p-4">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>
            <div className="card-kicker">TỔNG SỐ ĐÃ QUÉT</div>
            <div className="font-heading text-3xl font-bold">{sessionOverview.total}</div>
          </div>

          <div className="blueprint p-4">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>
            <div className="card-kicker">TỶ LỆ TIÊU CỰC</div>
            <div className="font-heading text-3xl font-bold text-[var(--color-negative)]">
              {sessionOverview.negPct}%
            </div>
          </div>

          <div className="blueprint p-4">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>
            <div className="card-kicker">KHÍA CẠNH BỊ CHÊ NHIỀU NHẤT</div>
            <div className="font-heading text-2xl font-bold truncate">{sessionOverview.topAspect}</div>
          </div>

          <div className="blueprint p-4">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>
            <div className="card-kicker">ĐỘ TIN CẬY TRUNG BÌNH</div>
            <div className="font-heading text-3xl font-bold">{sessionOverview.avgConfidence}%</div>
          </div>
        </div>

        {/* Donut Chart & Session Trend */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_1.4fr] gap-5">
          {/* Donut Card */}
          <div className="blueprint p-5 flex items-center gap-6">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>

            <svg
              width="120"
              height="120"
              viewBox="0 0 120 120"
              className="shrink-0 -rotate-90"
            >
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke="var(--color-neutral-200)"
                strokeWidth="14"
              />
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke="var(--color-accent-700)"
                strokeWidth="14"
                strokeDasharray={sessionOverview.donut.posDash}
                strokeDashoffset={sessionOverview.donut.posOffset}
              />
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke="var(--color-neutral-500)"
                strokeWidth="14"
                strokeDasharray={sessionOverview.donut.neuDash}
                strokeDashoffset={sessionOverview.donut.neuOffset}
              />
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke="var(--color-negative)"
                strokeWidth="14"
                strokeDasharray={sessionOverview.donut.negDash}
                strokeDashoffset={sessionOverview.donut.negOffset}
              />
            </svg>

            <div className="flex flex-col gap-2">
              <div className="card-kicker">TỶ LỆ CẢM XÚC TỔNG THỂ</div>
              <div className="flex flex-col gap-1.5 text-xs font-medium">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[var(--color-accent-700)]"></span>
                  Tích cực — <b>{sessionOverview.posPct}%</b>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[var(--color-neutral-500)]"></span>
                  Trung lập — <b>{sessionOverview.neuPct}%</b>
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[var(--color-negative)]"></span>
                  Tiêu cực — <b>{sessionOverview.negPct}%</b>
                </span>
              </div>
            </div>
          </div>

          {/* Sparkline Trend Card */}
          <div className="blueprint p-5 flex flex-col justify-between">
            <i className="corner tl"></i>
            <i className="corner tr"></i>
            <i className="corner bl"></i>
            <i className="corner br"></i>

            <div>
              <div className="card-kicker">XU HƯỚNG TỶ LỆ TIÊU CỰC TRONG PHIÊN</div>
              <div className="flex items-baseline gap-2 mb-2">
                <span className="card-title text-base">
                  {sessionOverview.endPct <= sessionOverview.startPct ? "Đang giảm dần" : "Đang tăng dần"}
                </span>
                <span className="text-xs text-[var(--color-accent-700)] font-semibold">
                  {sessionOverview.endPct <= sessionOverview.startPct ? "↓" : "↑"}{" "}
                  {sessionOverview.deltaAbs} điểm so với đầu phiên
                </span>
              </div>
            </div>

            <svg viewBox="0 0 300 100" width="100%" height="80" preserveAspectRatio="none">
              <polyline
                points={sessionOverview.sparklinePoints}
                fill="none"
                stroke="var(--color-negative)"
                strokeWidth="2"
              />
            </svg>

            <div className="flex justify-between text-[11px] opacity-60 font-medium">
              <span>Đầu phiên · {sessionOverview.startPct}%</span>
              <span>Hiện tại · {sessionOverview.endPct}%</span>
            </div>
          </div>
        </div>

        {/* Aspect Negative Ranking */}
        <div className="blueprint p-5 flex flex-col gap-3">
          <i className="corner tl"></i>
          <i className="corner tr"></i>
          <i className="corner bl"></i>
          <i className="corner br"></i>

          <div className="card-kicker">XẾP HẠNG KHÍA CẠNH THEO SỐ LẦN BỊ ĐÁNH GIÁ TIÊU CỰC</div>

          <div className="flex flex-col gap-2.5 mt-1">
            {sessionOverview.ranking.map((r, idx) => (
              <div key={idx} className="grid grid-cols-[160px_1fr_60px] items-center gap-3 text-xs">
                <span className="font-semibold truncate">{r.name}</span>
                <div className="h-2.5 bg-[var(--color-neutral-200)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--color-accent-700)] transition-all duration-300"
                    style={{ width: `${r.barPct}%` }}
                  ></div>
                </div>
                <span className="opacity-70 text-right font-mono">{r.count} lượt</span>
              </div>
            ))}
          </div>
        </div>

        {/* Detailed Session Log Table */}
        <div className="blueprint p-5 flex flex-col gap-4">
          <i className="corner tl"></i>
          <i className="corner tr"></i>
          <i className="corner bl"></i>
          <i className="corner br"></i>

          <div className="card-kicker">NHẬT KÝ CHI TIẾT</div>

          {/* Filter & Search Bar */}
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative max-w-sm flex-1">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 opacity-40" />
              <input
                className="input pl-8 text-xs"
                placeholder="Lọc theo từ khóa hoặc khía cạnh…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="seg">
              <span
                onClick={() => setSentimentFilter("all")}
                className={`seg-opt ${sentimentFilter === "all" ? "active" : ""}`}
              >
                Tất cả
              </span>
              <span
                onClick={() => setSentimentFilter("positive")}
                className={`seg-opt ${sentimentFilter === "positive" ? "active" : ""}`}
              >
                Tích cực
              </span>
              <span
                onClick={() => setSentimentFilter("neutral")}
                className={`seg-opt ${sentimentFilter === "neutral" ? "active" : ""}`}
              >
                Trung lập
              </span>
              <span
                onClick={() => setSentimentFilter("negative")}
                className={`seg-opt ${sentimentFilter === "negative" ? "active" : ""}`}
              >
                Tiêu cực
              </span>
            </div>
          </div>

          {/* Table */}
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th style={{ width: "80px" }}>Giờ</th>
                  <th style={{ width: "110px" }}>Cảm xúc</th>
                  <th style={{ width: "220px" }}>Khía cạnh tiêu cực</th>
                  <th>Nội dung review</th>
                </tr>
              </thead>
              <tbody>
                {filteredLog.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="text-center py-6 opacity-50">
                      Không tìm thấy kết quả phù hợp
                    </td>
                  </tr>
                ) : (
                  filteredLog.map((row, idx) => (
                    <tr key={idx}>
                      <td className="whitespace-nowrap opacity-70 font-mono text-xs">{row.time}</td>
                      <td>
                        <span
                          className="tag"
                          style={{ background: row.sentimentBg, color: row.sentimentColor }}
                        >
                          {row.sentimentLabel}
                        </span>
                      </td>
                      <td>
                        <div className="flex gap-1.5 flex-wrap">
                          {row.negAspects.length > 0 ? (
                            row.negAspects.map((na, i) => (
                              <span key={i} className="tag tag-neutral">
                                {na}
                              </span>
                            ))
                          ) : (
                            <span className="opacity-40 text-xs">—</span>
                          )}
                        </div>
                      </td>
                      <td className="opacity-85 text-xs max-w-md">{row.snippet}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

