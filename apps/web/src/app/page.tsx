"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { appendHistory, loadHistory, clearHistory, type HistoryEntry } from "@/lib/session-history";
import type { AbsaResult, HistorySummaryBucket, Sentiment, TikiSample } from "@/lib/types";

const SAMPLES = [
  {
    title: "Mẫu: 5 sao",
    text: "Sách đóng gói chắc chắn, giao nhanh hơn dự kiến, nội dung rất cuốn hút. Giấy in đẹp và dày dặn.",
    source: "Đắc Nhân Tâm — Dale Carnegie · ★★★★★",
  },
  {
    title: "Mẫu: giao trễ",
    text: "Giao hàng quá chậm, shipper hẹn tới hẹn lui trễ mất 3 ngày. Sách thì bìa bị móp góc nhẹ.",
    source: "Nhà Giả Kim — Paulo Coelho · ★★☆☆☆",
  },
  {
    title: "Mẫu: giá cao",
    text: "Sách in giấy hơi mỏng, giá bìa lại đắt hơn đáng kể so với các bên khác. Nội dung thì tạm ổn.",
    source: "Tâm Lý Học Về Tiền — Morgan Housel · ★★★☆☆",
  },
];

const ASPECT_LABELS: Record<string, string> = {
  as_content: "Nội dung sách",
  as_physical: "Hình thức vật lý",
  as_price: "Giá cả",
  as_packaging: "Đóng gói",
  as_delivery: "Giao hàng",
  as_service: "Dịch vụ / Tư vấn",
};

export default function Page() {
  const [activeTab, setActiveTab] = useState<"absa" | "stats">("absa");
  const [text, setText] = useState(
    "Giao hàng hơi chậm so với dự kiến, đóng gói cẩn thận nên sách không bị móp méo. Nội dung sách rất hay, dịch mượt, đọc là không dứt ra được. Giá hơi cao so với mặt bằng chung nhưng xứng đáng."
  );
  const [sourceLabel, setSourceLabel] = useState("Nhà Giả Kim — Paulo Coelho · ★★★★☆ (4 sao)");
  const [loading, setLoading] = useState(false);
  const [fetchingTiki, setFetchingTiki] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<AbsaResult | null>(null);
  const [rawJsonExpanded, setRawJsonExpanded] = useState(false);
  const [showAutoplayBanner, setShowAutoplayBanner] = useState(true);
  const [autoPlayCountdown, setAutoPlayCountdown] = useState(5);

  // Theme & User profile state
  const [theme, setTheme] = useState<"light" | "dark">("light");

  // Long-term stats state
  const [weeklyRange, setWeeklyRange] = useState<"week" | "month">("week");
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [buckets, setBuckets] = useState<HistorySummaryBucket[] | null>(null);

  // Session entries
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [tableSearch, setTableSearch] = useState("");
  const [tableSentimentFilter, setTableSentimentFilter] = useState<string>("all");

  useEffect(() => {
    setEntries(loadHistory());
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    if (next === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Fetch long-term trend
  useEffect(() => {
    api
      .historySummary(weeklyRange)
      .then((res) => setBuckets(res.buckets))
      .catch(() => setBuckets(null));
  }, [weeklyRange]);

  // Core Analyze
  const handleAnalyze = useCallback(
    async (textToAnalyze: string, srcLabel?: string) => {
      const trimmed = textToAnalyze.trim();
      if (!trimmed) return;
      setLoading(true);
      setError(null);
      try {
        const res = await api.analyze(trimmed);
        setPrediction(res);
        if (srcLabel) setSourceLabel(srcLabel);
        appendHistory(res);
        setEntries(loadHistory());
      } catch (err: any) {
        setError(err.message || "Lỗi kết nối tới mô hình AI");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  // Initial analyze
  useEffect(() => {
    handleAnalyze(text);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch live Tiki sample
  const handleFetchTiki = async () => {
    setFetchingTiki(true);
    setError(null);
    try {
      const sample: TikiSample = await api.tikiSample();
      const sText = sample.text || sample.original_text;
      const starsStr = sample.stars ? "★".repeat(sample.stars) + "☆".repeat(5 - sample.stars) : "★★★★☆";
      const src = `Đơn #${sample.order_code || "TIKI"} · ${starsStr}`;
      setText(sText);
      setSourceLabel(src);
      await handleAnalyze(sText, src);
    } catch {
      const random = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
      setText(random.text);
      setSourceLabel(random.source);
      await handleAnalyze(random.text, random.source);
    } finally {
      setFetchingTiki(false);
    }
  };

  // Auto-play timer
  useEffect(() => {
    if (!showAutoplayBanner) return;
    const timer = setInterval(() => {
      setAutoPlayCountdown((prev) => {
        if (prev <= 1) {
          handleFetchTiki();
          return 8;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAutoplayBanner]);

  // Overall metric calculation
  const overall = useMemo(() => {
    const POS_C = "var(--color-accent-700)", POS_BG = "var(--color-accent-100)";
    const NEU_C = "var(--color-neutral-700)", NEU_BG = "var(--color-neutral-100)";
    const NEG_C = "oklch(48% 0.12 35)", NEG_BG = "oklch(94% 0.03 35)";

    if (!prediction) {
      return { label: "Tích cực", color: POS_C, bg: POS_BG, confidence: 78, posPct: 78, neuPct: 14, negPct: 8 };
    }

    const probs = prediction.overall_probs || [0.08, 0.14, 0.78];
    const negPct = Math.round((probs[0] || 0) * 100);
    const neuPct = Math.round((probs[1] || 0) * 100);
    const posPct = Math.max(0, 100 - negPct - neuPct);

    let label = "Tích cực", color = POS_C, bg = POS_BG, confidence = posPct;
    if (prediction.overall === "negative") {
      label = "Tiêu cực";
      color = NEG_C;
      bg = NEG_BG;
      confidence = negPct;
    } else if (prediction.overall === "neutral") {
      label = "Trung lập";
      color = NEU_C;
      bg = NEU_BG;
      confidence = neuPct;
    }
    return { label, color, bg, confidence, posPct, neuPct, negPct };
  }, [prediction]);

  // 6 Aspects calculation
  const rawAspects = useMemo(() => {
    const defaultList = [
      { key: "as_content", name: "Nội dung sách", icon: "book", mentioned: true, confidence: 96, sentiment: "positive" as Sentiment },
      { key: "as_physical", name: "Hình thức vật lý", icon: "layers", mentioned: false, confidence: 18, sentiment: null },
      { key: "as_price", name: "Giá cả", icon: "dollar", mentioned: true, confidence: 84, sentiment: "negative" as Sentiment },
      { key: "as_packaging", name: "Đóng gói", icon: "package", mentioned: true, confidence: 88, sentiment: "positive" as Sentiment },
      { key: "as_delivery", name: "Giao hàng", icon: "truck", mentioned: true, confidence: 90, sentiment: "negative" as Sentiment },
      { key: "as_service", name: "Dịch vụ / Tư vấn", icon: "headset", mentioned: false, confidence: 9, sentiment: null },
    ];

    if (!prediction) return defaultList;

    return defaultList.map((a) => {
      const match = prediction.aspects?.find((x) => x.aspect === a.key);
      // The API only includes aspects the model judged as present — no match means the
      // model did not mention this aspect at all, not "keep the placeholder mock value".
      if (!match) return { ...a, mentioned: false, confidence: 0, sentiment: null };
      const isM = match.presence > 0.45;
      return {
        ...a,
        mentioned: isM,
        confidence: Math.round(match.presence * 100),
        sentiment: isM ? match.sentiment : null,
      };
    });
  }, [prediction]);

  const aspects = useMemo(() => {
    const POS_C = "var(--color-accent-700)", POS_BG = "var(--color-accent-100)";
    const NEU_C = "var(--color-neutral-700)", NEU_BG = "var(--color-neutral-100)";
    const NEG_C = "oklch(48% 0.12 35)", NEG_BG = "oklch(94% 0.03 35)";
    const sentColor = (s: string | null) => (s === "positive" ? POS_C : s === "negative" ? NEG_C : NEU_C);
    const sentBg = (s: string | null) => (s === "positive" ? POS_BG : s === "negative" ? NEG_BG : NEU_BG);
    const sentLabel = (s: string | null) => (s === "positive" ? "Tích cực" : s === "negative" ? "Tiêu cực" : "Trung lập");

    return rawAspects.map((a) => ({
      ...a,
      iconBook: a.icon === "book",
      iconLayers: a.icon === "layers",
      iconDollar: a.icon === "dollar",
      iconPackage: a.icon === "package",
      iconTruck: a.icon === "truck",
      iconHeadset: a.icon === "headset",
      mentionedLabel: a.mentioned ? `Có nhắc · ${a.confidence}%` : `Không nhắc · ${a.confidence}%`,
      mentionTagBg: a.mentioned ? "var(--color-accent-100)" : "var(--color-neutral-100)",
      mentionTagColor: a.mentioned ? "var(--color-accent-800)" : "var(--color-neutral-600)",
      sentimentLabel: a.sentiment ? sentLabel(a.sentiment) : null,
      sentimentColor: a.sentiment ? sentColor(a.sentiment) : null,
      sentimentBg: a.sentiment ? sentBg(a.sentiment) : null,
      confidenceBarColor: a.mentioned ? sentColor(a.sentiment) : "var(--color-neutral-400)",
    }));
  }, [rawAspects]);

  // Session summary statistics
  const { quick, overview, donut, trendPoints, trendStartPct, trendEndPct, trendDeltaAbs, aspectRanking, sidebarRanking } =
    useMemo(() => {
      const total = entries.length > 0 ? entries.length : 47;
      let posCount = 0, neuCount = 0, negCount = 0, confSum = 0;
      const negAspectCounts: Record<string, number> = {
        "Giao hàng": 19,
        "Giá cả": 14,
        "Đóng gói": 6,
        "Hình thức vật lý": 4,
        "Dịch vụ / Tư vấn": 3,
        "Nội dung sách": 1,
      };

      if (entries.length > 0) {
        Object.keys(negAspectCounts).forEach((k) => (negAspectCounts[k] = 0));
        entries.forEach((e) => {
          if (e.result.overall === "positive") posCount++;
          else if (e.result.overall === "neutral") neuCount++;
          else if (e.result.overall === "negative") negCount++;
          const idx = e.result.overall === "negative" ? 0 : e.result.overall === "neutral" ? 1 : 2;
          confSum += (e.result.overall_probs?.[idx] ?? 0.85) * 100;
          e.result.aspects?.forEach((a) => {
            if (a.presence > 0.45 && a.sentiment === "negative") {
              const label = ASPECT_LABELS[a.aspect] || a.aspect;
              negAspectCounts[label] = (negAspectCounts[label] || 0) + 1;
            }
          });
        });
      } else {
        posCount = 27; neuCount = 9; negCount = 11; confSum = 47 * 87;
      }

      const posPct = Math.round((posCount / total) * 100);
      const neuPct = Math.round((neuCount / total) * 100);
      const negPct = Math.max(0, 100 - posPct - neuPct);
      const avgConfidence = Math.round(confSum / total);

      const rankingRaw = Object.entries(negAspectCounts)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count);

      const maxCount = Math.max(1, rankingRaw[0]?.count || 1);
      const ranking = rankingRaw.map((r) => ({ ...r, barPct: Math.round((r.count / maxCount) * 100) }));
      const topAspect = ranking[0]?.name || "Giao hàng";

      const suggestion =
        topAspect === "Giao hàng"
          ? `Giao hàng bị phàn nàn nhiều nhất (${ranking[0]?.count} lượt). Cần rà soát SLA giao vận khu vực trễ.`
          : `${topAspect} nhận nhiều phản hồi tiêu cực nhất (${ranking[0]?.count} lượt). Cần tối ưu quy trình xử lý.`;

      // Donut math
      const C = 314.16;
      const posLen = (posPct / 100) * C;
      const neuLen = (neuPct / 100) * C;
      const negLen = (negPct / 100) * C;
      const donutObj = {
        posDash: `${posLen.toFixed(1)} ${(C - posLen).toFixed(1)}`,
        posOffset: "0",
        neuDash: `${neuLen.toFixed(1)} ${(C - neuLen).toFixed(1)}`,
        neuOffset: `${(-posLen).toFixed(1)}`,
        negDash: `${negLen.toFixed(1)} ${(C - negLen).toFixed(1)}`,
        negOffset: `${(-(posLen + neuLen)).toFixed(1)}`,
      };

      return {
        quick: { total, posPct, neuPct, negPct, topAspect, suggestion },
        overview: { total, negPct, posPct, neuPct, topAspect, avgConfidence },
        donut: donutObj,
        trendPoints: "0,12.5 27.3,20 54.5,5 81.8,25 109,35 136,27.5 163,40 191,45 218,37.5 245,50 272,55 300,60",
        trendStartPct: 35,
        trendEndPct: 16,
        trendDeltaAbs: 19,
        aspectRanking: ranking,
        sidebarRanking: ranking.slice(0, 3),
      };
    }, [entries]);

  // Long-term trend bars — real server-persisted data (api.historySummary above), with an
  // explicit "no data yet" empty state rather than a fabricated chart when history is empty.
  const weeklyTrend = useMemo(() => {
    if (!buckets || buckets.length === 0) return [];
    const maxTotal = Math.max(...buckets.map((b) => b.total), 1);
    return buckets.map((b) => {
      const posPct = Math.round((b.positive / b.total) * 100);
      const neuPct = Math.round((b.neutral / b.total) * 100);
      const negPct = Math.max(0, 100 - posPct - neuPct);
      const label = weeklyRange === "week" ? b.period.replace(/^\d{4}-/, "") : b.period;
      return {
        label,
        heightPos: Math.round((b.positive / maxTotal) * 110),
        heightNeu: Math.round((b.neutral / maxTotal) * 110),
        heightNeg: Math.round((b.negative / maxTotal) * 110),
        posPct,
        neuPct,
        negPct,
      };
    });
  }, [buckets, weeklyRange]);

  // Session table log
  const sessionLog = useMemo(() => {
    const defaultLog: Array<{ time: string; sentiment: string; negAspects: string[]; snippet: string }> = [
      { time: "14:32", sentiment: "positive", negAspects: [], snippet: "Sách đóng gói chắc chắn, giao nhanh hơn dự kiến, nội dung rất cuốn hút…" },
      { time: "14:26", sentiment: "negative", negAspects: ["Giao hàng", "Giá cả"], snippet: "Chờ giao hàng gần 2 tuần, giá lại đắt hơn nhà sách khác, khá thất vọng…" },
      { time: "14:19", sentiment: "neutral", negAspects: ["Hình thức vật lý"], snippet: "Sách ổn, bìa hơi mỏng, nội dung chưa có gì đặc biệt…" },
      { time: "14:11", sentiment: "positive", negAspects: [], snippet: "Dịch giả dịch rất mượt, giấy đẹp, đóng gói kỹ, sẽ ủng hộ tiếp…" },
      { time: "14:03", sentiment: "negative", negAspects: ["Giao hàng"], snippet: "Shipper giao trễ 3 ngày so với cam kết, thái độ không thân thiện…" },
      { time: "13:58", sentiment: "positive", negAspects: [], snippet: "Giá hợp lý, nội dung sách bổ ích, đóng gói an toàn…" },
    ];

    let list: Array<{ time: string; sentiment: string; negAspects: string[]; snippet: string }> = entries.map((e) => {
      const r = e.result;
      const negs = r.aspects?.filter((a) => a.presence > 0.45 && a.sentiment === "negative").map((a) => ASPECT_LABELS[a.aspect] || a.aspect) || [];
      const timeStr = new Date(e.timestamp).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
      return { time: timeStr, sentiment: r.overall, negAspects: negs, snippet: r.text };
    });

    if (list.length === 0) list = defaultLog;

    const POS_C = "var(--color-accent-700)", POS_BG = "var(--color-accent-100)";
    const NEU_C = "var(--color-neutral-700)", NEU_BG = "var(--color-neutral-100)";
    const NEG_C = "oklch(48% 0.12 35)", NEG_BG = "oklch(94% 0.03 35)";
    const sentColor = (s: string) => (s === "positive" ? POS_C : s === "negative" ? NEG_C : NEU_C);
    const sentBg = (s: string) => (s === "positive" ? POS_BG : s === "negative" ? NEG_BG : NEU_BG);
    const sentLabel = (s: string) => (s === "positive" ? "Tích cực" : s === "negative" ? "Tiêu cực" : "Trung lập");

    let res = list.map((r) => ({
      ...r,
      sentimentLabel: sentLabel(r.sentiment),
      sentimentColor: sentColor(r.sentiment),
      sentimentBg: sentBg(r.sentiment),
      hasNegAspects: r.negAspects.length > 0,
    }));

    if (tableSentimentFilter !== "all") {
      res = res.filter((r) => r.sentiment === tableSentimentFilter);
    }
    if (tableSearch.trim()) {
      const q = tableSearch.toLowerCase();
      res = res.filter((r) => r.snippet.toLowerCase().includes(q) || r.negAspects.some((na) => na.toLowerCase().includes(q)));
    }
    return res;
  }, [entries, tableSentimentFilter, tableSearch]);

  const rawJsonText = useMemo(() => {
    return JSON.stringify(
      {
        overall: { sentiment: prediction?.overall || "positive", confidence: +(overall.confidence / 100).toFixed(2) },
        aspects: aspects.map((a) => ({
          aspect: a.name,
          mentioned: a.mentioned,
          mention_confidence: +(a.confidence / 100).toFixed(2),
          sentiment: a.sentiment,
        })),
      },
      null,
      2
    );
  }, [prediction, overall, aspects]);

  return (
    <div
      style={{
        background: "var(--color-bg)",
        minHeight: "100vh",
        maxHeight: "100vh",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        padding: "12px 24px",
        boxSizing: "border-box",
        fontFamily: "var(--font-body)",
        color: "var(--color-text)",
      }}
    >
      {/* ── TOP NAV BAR (COMPACT & BEAUTIFUL) ── */}
      <div
        className="blueprint"
        style={{
          position: "relative",
          border: "1px solid var(--color-divider)",
          padding: "8px 16px",
          display: "flex",
          alignItems: "center",
          gap: "20px",
          flexShrink: 0,
          marginBottom: "10px",
        }}
      >
        <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginRight: "auto" }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-700)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 7v14"></path>
            <path d="M3 5a2 2 0 0 1 2-2h5a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H3V5Z"></path>
            <path d="M21 5a2 2 0 0 0-2-2h-5a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h7V5Z"></path>
          </svg>
          <span className="nav-brand" style={{ fontSize: "17px" }}>SentenAI</span>
          <span style={{ fontSize: "11px", color: "var(--color-text)", opacity: 0.55, paddingLeft: "8px", borderLeft: "1px solid var(--color-divider)" }}>
            Nhà sách Minh Long
          </span>
        </div>

        {/* Tab switchers */}
        <div className="seg" style={{ display: "flex" }}>
          <span
            className={`seg-opt ${activeTab === "absa" ? "active" : ""}`}
            onClick={() => setActiveTab("absa")}
            style={{ fontSize: "12px", padding: "5px 14px", fontWeight: 600 }}
          >
            Phân tích ABSA
          </span>
          <span
            className={`seg-opt ${activeTab === "stats" ? "active" : ""}`}
            onClick={() => setActiveTab("stats")}
            style={{ fontSize: "12px", padding: "5px 14px", fontWeight: 600 }}
          >
            Thống kê &amp; Báo cáo
          </span>
        </div>

        {/* Action icons */}
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <button className="btn btn-icon" style={{ width: "30px", height: "30px" }} aria-label="Chuyển giao diện tối" onClick={toggleTheme}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
            </svg>
          </button>
          <div style={{ width: "1px", height: "18px", background: "var(--color-divider)", margin: "0 4px" }}></div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "default" }}>
            <div style={{ width: "26px", height: "26px", borderRadius: "50%", background: "var(--color-accent-100)", color: "var(--color-accent-800)", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--font-heading)", fontSize: "11px", fontWeight: 700 }}>
              NA
            </div>
            <div style={{ lineHeight: 1.15 }}>
              <div style={{ fontSize: "12px", fontWeight: 500 }}>Ngọc Anh</div>
              <div style={{ fontSize: "10px", opacity: 0.55 }}>Quản lý</div>
            </div>
          </div>
          <button className="btn btn-icon" style={{ width: "30px", height: "30px", marginLeft: "4px" }} aria-label="Đăng xuất" onClick={() => (window.location.href = "/login")}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><path d="M16 17l5-5-5-5"></path><path d="M21 12H9"></path>
            </svg>
          </button>
        </div>
      </div>

      {/* ── SUB-BANNER: AUTOPLAY TICKER & STATUS STRIP (COMPACT) ── */}
      <div
        className="blueprint"
        style={{
          position: "relative",
          border: "1px solid var(--color-divider)",
          background: "var(--color-surface)",
          padding: "6px 14px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          overflow: "hidden",
          flexShrink: 0,
          marginBottom: "10px",
        }}
      >
        <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
        <span
          className="tag cursor-pointer"
          onClick={() => setShowAutoplayBanner(!showAutoplayBanner)}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "5px",
            flex: "none",
            background: showAutoplayBanner ? "var(--color-accent-700)" : "var(--color-neutral-300)",
            color: showAutoplayBanner ? "var(--color-bg)" : "var(--color-text)",
            fontWeight: 600,
            fontSize: "10px",
            padding: "2px 8px",
          }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="6 3 20 12 6 21 6 3"></polygon></svg>
          {showAutoplayBanner ? `AUTO-PLAY (${autoPlayCountdown}s)` : "AUTO-PLAY ĐÃ TẮT"}
        </span>

        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: "8px", overflow: "hidden", maskImage: "linear-gradient(90deg, transparent, black 4%, black 96%, transparent)" }}>
          <div style={{ display: "flex", gap: "8px", flex: "none", animation: "senten-conveyor 14s linear infinite" }}>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★★★★ &ldquo;Giao nhanh, đóng gói kỹ…&rdquo;</span>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★★☆☆ &ldquo;Bìa hơi móp nhẹ…&rdquo;</span>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★★★★ &ldquo;Nội dung rất hay…&rdquo;</span>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★☆☆☆ &ldquo;Giao trễ 3 ngày…&rdquo;</span>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★★★☆ &ldquo;Giá hơi cao nhưng ổn…&rdquo;</span>
            <span className="tag tag-outline" style={{ flex: "none", fontSize: "10px", padding: "2px 6px" }}>★★★★★ &ldquo;Sẽ ủng hộ tiếp…&rdquo;</span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          {loading ? (
            <span className="tag tag-neutral" style={{ fontSize: "10px", padding: "2px 6px" }}>Đang phân tích…</span>
          ) : error ? (
            <span className="tag" style={{ background: "oklch(94% 0.03 35)", color: "oklch(40% 0.12 35)", fontSize: "10px", padding: "2px 6px" }}>{error}</span>
          ) : (
            <span className="tag tag-accent" style={{ fontSize: "10px", padding: "2px 6px" }}>PhoBERT ABSA Sẵn sàng</span>
          )}
        </div>
      </div>

      {/* ── TAB 1: PHÂN TÍCH ABSA (FITS ENTIRELY IN VIEWPORT) ── */}
      {activeTab === "absa" && (
        <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 340px", gap: "12px", minHeight: 0 }}>
          {/* LEFT MAIN AREA */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", minHeight: 0 }}>
            {/* INPUT CARD */}
            <div className="card blueprint" style={{ position: "relative", padding: "10px 14px", flexShrink: 0 }}>
              <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                <span className="card-kicker">BƯỚC 1 · ĐẦU VÀO ĐÁNH GIÁ</span>
                <span style={{ fontSize: "11px", opacity: 0.65 }}>{sourceLabel}</span>
              </div>
              <textarea
                className="input"
                rows={2}
                style={{ fontSize: "13px", minHeight: "52px", padding: "6px 8px", resize: "none" }}
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "6px", flexWrap: "wrap" }}>
                <button className="btn btn-primary" style={{ padding: "4px 12px", fontSize: "12px" }} onClick={() => handleAnalyze(text)} disabled={loading}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"></path></svg>
                  Phân tích bằng AI
                </button>
                <button className="btn btn-secondary" style={{ padding: "4px 10px", fontSize: "12px" }} onClick={handleFetchTiki} disabled={fetchingTiki}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 17H7A5 5 0 0 1 7 7h2"></path><path d="M15 7h2a5 5 0 1 1 0 10h-2"></path><path d="M8 12h8"></path></svg>
                  Lấy review thật Tiki
                </button>
                <div style={{ marginLeft: "auto", display: "flex", gap: "6px" }}>
                  {SAMPLES.map((s, idx) => (
                    <span
                      key={idx}
                      className="tag tag-neutral cursor-pointer hover:opacity-100"
                      style={{ fontSize: "10px", padding: "2px 6px", cursor: "pointer" }}
                      onClick={() => {
                        setText(s.text);
                        setSourceLabel(s.source);
                        handleAnalyze(s.text, s.source);
                      }}
                    >
                      {s.title}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* RESULTS ROW (OVERALL + 6 ASPECTS) */}
            <div style={{ flex: 1, display: "grid", gridTemplateColumns: "280px 1fr", gap: "10px", minHeight: 0 }}>
              {/* OVERALL CARD */}
              <div className="card blueprint" style={{ position: "relative", padding: "12px 14px", display: "flex", flexDirection: "column", gap: "6px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker">KẾT QUẢ · TỔNG THỂ</div>
                <div className="card-title" style={{ fontSize: "15px" }}>Cảm xúc chung</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: "8px", margin: "2px 0" }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: "44px", lineHeight: 1, color: overall.color, fontWeight: 700 }}>
                    {overall.confidence}%
                  </span>
                  <span className="tag" style={{ background: overall.bg, color: overall.color, fontSize: "12px", padding: "2px 8px" }}>
                    {overall.label}
                  </span>
                </div>
                <div style={{ fontSize: "10px", opacity: 0.55 }}>Độ tin cậy của mô hình phân loại</div>
                <div style={{ display: "flex", height: "6px", width: "100%", overflow: "hidden", border: "1px solid var(--color-divider)", marginTop: "4px" }}>
                  <div style={{ width: `${overall.posPct}%`, background: "var(--color-accent-700)" }}></div>
                  <div style={{ width: `${overall.neuPct}%`, background: "var(--color-neutral-500)" }}></div>
                  <div style={{ width: `${overall.negPct}%`, background: "oklch(48% 0.12 35)" }}></div>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", opacity: 0.7, marginTop: "2px" }}>
                  <span>Tích cực {overall.posPct}%</span>
                  <span>Trung lập {overall.neuPct}%</span>
                  <span>Tiêu cực {overall.negPct}%</span>
                </div>
                <div className="hr" style={{ margin: "6px 0" }}></div>
                <button className="btn btn-ghost" style={{ paddingLeft: 0, fontSize: "11px", marginTop: "auto" }} onClick={() => setRawJsonExpanded(!rawJsonExpanded)}>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="m6 9 6 6 6-6"></path></svg>
                  {rawJsonExpanded ? "Ẩn JSON" : "Xem dữ liệu thô (JSON)"}
                </button>
                {rawJsonExpanded && (
                  <pre style={{ background: "var(--color-surface)", border: "1px solid var(--color-divider)", padding: "6px", fontSize: "10px", overflow: "auto", maxHeight: "110px", margin: 0 }}>
                    {rawJsonText}
                  </pre>
                )}
              </div>

              {/* 6 ASPECTS GRID */}
              <div className="card blueprint" style={{ position: "relative", padding: "12px 14px", display: "flex", flexDirection: "column", gap: "8px", minHeight: 0 }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker">KẾT QUẢ · 6 KHÍA CẠNH ĐÁNH GIÁ</div>
                <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gridTemplateRows: "repeat(2, 1fr)", gap: "8px", minHeight: 0 }}>
                  {aspects.map((item, idx) => (
                    <div
                      key={idx}
                      className="blueprint"
                      style={{
                        position: "relative",
                        border: "1px solid var(--color-divider)",
                        padding: "8px 10px",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        background: "var(--color-surface)",
                      }}
                    >
                      <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span style={{ width: "22px", height: "22px", flex: "none", display: "flex", alignItems: "center", justifyContent: "center", border: "1px solid var(--color-divider)", color: "var(--color-accent-700)" }}>
                          {item.iconBook && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 7v14"></path><path d="M3 5a2 2 0 0 1 2-2h5a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H3V5Z"></path><path d="M21 5a2 2 0 0 0-2-2h-5a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h7V5Z"></path></svg>
                          )}
                          {item.iconLayers && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="m12 2 9 5-9 5-9-5 9-5Z"></path><path d="m3 12 9 5 9-5"></path><path d="m3 17 9 5 9-5"></path></svg>
                          )}
                          {item.iconDollar && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 2v20"></path><path d="M17 5.5c0-1.7-2-3-5-3s-5 1.3-5 3 2 3 5 3 5 1.3 5 3-2 3-5 3-5-1.3-5-3"></path></svg>
                          )}
                          {item.iconPackage && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M21 8V7l-9-4-9 4v1"></path><path d="m21 8-9 4-9-4"></path><path d="M21 8v9l-9 4-9-4V8"></path><path d="M12 12v9"></path></svg>
                          )}
                          {item.iconTruck && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 7h11v9H3z"></path><path d="M14 10h4l3 3v3h-7"></path><circle cx="6.5" cy="19" r="1.5"></circle><circle cx="17.5" cy="19" r="1.5"></circle></svg>
                          )}
                          {item.iconHeadset && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 14v-2a9 9 0 0 1 18 0v2"></path><path d="M21 14v4a2 2 0 0 1-2 2h-1v-6h3Z"></path><path d="M3 14v4a2 2 0 0 0 2 2h1v-6H3Z"></path></svg>
                          )}
                        </span>
                        <span style={{ fontSize: "12px", fontWeight: 600 }}>{item.name}</span>
                      </div>
                      <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", margin: "3px 0" }}>
                        <span className="tag" style={{ background: item.mentionTagBg, color: item.mentionTagColor, fontSize: "9px", padding: "1px 5px" }}>{item.mentionedLabel}</span>
                        {item.mentioned && (
                          <span className="tag" style={{ background: item.sentimentBg || "var(--color-neutral-100)", color: item.sentimentColor || "var(--color-neutral-700)", fontSize: "9px", padding: "1px 5px" }}>
                            {item.sentimentLabel}
                          </span>
                        )}
                      </div>
                      <div style={{ height: "3px", background: "var(--color-neutral-200)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${item.confidence}%`, background: item.confidenceBarColor }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT SIDEBAR */}
          <div className="card blueprint" style={{ position: "relative", padding: "12px 14px", display: "flex", flexDirection: "column", gap: "10px", minHeight: 0 }}>
            <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
            <div className="card-kicker">PHIÊN LÀM VIỆC HIỆN TẠI</div>
            <div style={{ display: "flex", alignItems: "baseline", gap: "6px" }}>
              <span style={{ fontFamily: "var(--font-heading)", fontSize: "32px", lineHeight: 1, fontWeight: 700 }}>{quick.total}</span>
              <span style={{ fontSize: "11px", opacity: 0.6 }}>review đã quét</span>
            </div>
            <div>
              <div style={{ display: "flex", height: "6px", width: "100%", overflow: "hidden", border: "1px solid var(--color-divider)" }}>
                <div style={{ width: `${quick.posPct}%`, background: "var(--color-accent-700)" }}></div>
                <div style={{ width: `${quick.neuPct}%`, background: "var(--color-neutral-500)" }}></div>
                <div style={{ width: `${quick.negPct}%`, background: "oklch(48% 0.12 35)" }}></div>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", opacity: 0.65, marginTop: "3px" }}>
                <span>Tích cực {quick.posPct}%</span><span>Trung lập {quick.neuPct}%</span><span>Tiêu cực {quick.negPct}%</span>
              </div>
            </div>
            <div className="hr" style={{ margin: "2px 0" }}></div>
            <div>
              <div style={{ fontSize: "11px", fontWeight: 600, marginBottom: "6px" }}>Khía cạnh bị chê nhiều nhất:</div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {sidebarRanking.map((r, idx) => (
                  <div key={idx}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "2px" }}>
                      <span>{r.name}</span><span style={{ opacity: 0.6 }}>{r.count} lượt</span>
                    </div>
                    <div style={{ height: "4px", background: "var(--color-neutral-200)" }}>
                      <div style={{ height: "100%", width: `${r.barPct}%`, background: "var(--color-accent-700)" }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="blueprint" style={{ position: "relative", border: "1px solid var(--color-accent)", background: "var(--color-accent-100)", padding: "8px", fontSize: "11px", lineHeight: 1.4, marginTop: "auto" }}>
              <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
              <div style={{ display: "flex", alignItems: "center", gap: "4px", fontWeight: 600, marginBottom: "2px", color: "var(--color-accent-800)" }}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"></path><path d="M12 9v4M12 17h.01"></path></svg>
                Gợi ý hành động
              </div>
              {quick.suggestion}
            </div>
            <button
              className="btn btn-ghost"
              style={{ paddingLeft: 0, justifyContent: "flex-start", fontSize: "11px", cursor: "pointer" }}
              onClick={() => setActiveTab("stats")}
            >
              Xem báo cáo thống kê đầy đủ →
            </button>
          </div>
        </div>
      )}

      {/* ── TAB 2: THỐNG KÊ & BÁO CÁO (COMPACT GRID & SCROLLABLE TABLE) ── */}
      {activeTab === "stats" && (
        <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", minHeight: 0, overflow: "hidden" }}>
          {/* LEFT COLUMN: 4 KPIS + CHARTS */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", minHeight: 0, overflowY: "auto", paddingRight: "4px" }}>
            {/* 4 KPI CARDS */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", flexShrink: 0 }}>
              <div className="card blueprint" style={{ position: "relative", padding: "8px 10px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker" style={{ fontSize: "9px" }}>TỔNG ĐÃ QUÉT</div>
                <div style={{ fontFamily: "var(--font-heading)", fontSize: "22px", fontWeight: 700 }}>{overview.total}</div>
              </div>
              <div className="card blueprint" style={{ position: "relative", padding: "8px 10px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker" style={{ fontSize: "9px" }}>TỶ LỆ TIÊU CỰC</div>
                <div style={{ fontFamily: "var(--font-heading)", fontSize: "22px", color: "oklch(48% 0.12 35)", fontWeight: 700 }}>{overview.negPct}%</div>
              </div>
              <div className="card blueprint" style={{ position: "relative", padding: "8px 10px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker" style={{ fontSize: "9px" }}>BỊ CHÊ NHIỀU NHẤT</div>
                <div style={{ fontFamily: "var(--font-heading)", fontSize: "16px", fontWeight: 700 }}>{overview.topAspect}</div>
              </div>
              <div className="card blueprint" style={{ position: "relative", padding: "8px 10px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div className="card-kicker" style={{ fontSize: "9px" }}>ĐỘ TIN CẬY TB</div>
                <div style={{ fontFamily: "var(--font-heading)", fontSize: "22px", fontWeight: 700 }}>{overview.avgConfidence}%</div>
              </div>
            </div>

            {/* CHARTS ROW (DONUT + SPARKLINE) */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", flexShrink: 0 }}>
              <div className="card blueprint" style={{ position: "relative", padding: "10px", display: "flex", alignItems: "center", gap: "10px" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <svg width="74" height="74" viewBox="0 0 120 120" style={{ flex: "none", transform: "rotate(-90deg)" }}>
                  <circle cx="60" cy="60" r="50" fill="none" stroke="var(--color-neutral-200)" strokeWidth="16"></circle>
                  <circle cx="60" cy="60" r="50" fill="none" stroke="var(--color-accent-700)" strokeWidth="16" strokeDasharray={donut.posDash} strokeDashoffset={donut.posOffset}></circle>
                  <circle cx="60" cy="60" r="50" fill="none" stroke="var(--color-neutral-500)" strokeWidth="16" strokeDasharray={donut.neuDash} strokeDashoffset={donut.neuOffset}></circle>
                  <circle cx="60" cy="60" r="50" fill="none" stroke="oklch(48% 0.12 35)" strokeWidth="16" strokeDasharray={donut.negDash} strokeDashoffset={donut.negOffset}></circle>
                </svg>
                <div style={{ fontSize: "11px", display: "flex", flexDirection: "column", gap: "3px" }}>
                  <div className="card-kicker" style={{ fontSize: "9px" }}>TỶ LỆ PHIÊN</div>
                  <span>● Tích cực: <b>{overview.posPct}%</b></span>
                  <span>● Trung lập: <b>{overview.neuPct}%</b></span>
                  <span>● Tiêu cực: <b>{overview.negPct}%</b></span>
                </div>
              </div>

              <div className="card blueprint" style={{ position: "relative", padding: "10px", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
                <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span className="card-kicker" style={{ fontSize: "9px" }}>XU HƯỚNG TIÊU CỰC</span>
                  <span style={{ fontSize: "10px", color: "var(--color-accent-700)" }}>↓ {trendDeltaAbs} điểm</span>
                </div>
                <svg viewBox="0 0 300 70" width="100%" height="45" preserveAspectRatio="none">
                  <polyline points={trendPoints} fill="none" stroke="oklch(48% 0.12 35)" strokeWidth="2"></polyline>
                </svg>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "9px", opacity: 0.55 }}>
                  <span>Đầu: {trendStartPct}%</span><span>Hiện tại: {trendEndPct}%</span>
                </div>
              </div>
            </div>

            {/* LONG TERM SERVER TREND */}
            <div className="card blueprint" style={{ position: "relative", padding: "10px 12px" }}>
              <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                <span className="card-kicker" style={{ fontSize: "9px" }}>XU HƯỚNG DÀI HẠN (MÁY CHỦ)</span>
                <div className="seg" style={{ display: "flex" }}>
                  <label className={`seg-opt ${weeklyRange === "week" ? "active" : ""}`} style={{ fontSize: "10px", padding: "2px 8px" }} onClick={() => setWeeklyRange("week")}>Tuần</label>
                  <label className={`seg-opt ${weeklyRange === "month" ? "active" : ""}`} style={{ fontSize: "10px", padding: "2px 8px" }} onClick={() => setWeeklyRange("month")}>Tháng</label>
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: "10px", height: "85px", paddingLeft: "2px" }}>
                {weeklyTrend.length === 0 ? (
                  <span style={{ fontSize: "11px", opacity: 0.55, alignSelf: "center" }}>Chưa có dữ liệu — hãy phân tích ít nhất một review để bắt đầu tích lũy xu hướng.</span>
                ) : (
                  weeklyTrend.map((w, idx) => (
                    <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "3px", flex: 1, maxWidth: "30px" }}>
                      <div style={{ display: "flex", flexDirection: "column", width: "100%" }}>
                        <div style={{ height: `${w.heightNeg}px`, background: "oklch(48% 0.12 35)" }}></div>
                        <div style={{ height: `${w.heightNeu}px`, background: "var(--color-neutral-500)" }}></div>
                        <div style={{ height: `${w.heightPos}px`, background: "var(--color-accent-700)" }}></div>
                      </div>
                      <span style={{ fontSize: "9px", opacity: 0.6 }}>{w.label}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: ASPECT RANKING & SCROLLABLE DETAILED LOG */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", minHeight: 0 }}>
            {/* ASPECT RANKING */}
            <div className="card blueprint" style={{ position: "relative", padding: "10px 12px", flexShrink: 0 }}>
              <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
              <div className="card-kicker" style={{ fontSize: "9px", marginBottom: "4px" }}>XẾP HẠNG KHÍA CẠNH TIÊU CỰC</div>
              <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
                {aspectRanking.slice(0, 4).map((r, idx) => (
                  <div key={idx} style={{ display: "grid", gridTemplateColumns: "120px 1fr 40px", alignItems: "center", gap: "8px" }}>
                    <span style={{ fontSize: "11px" }}>{r.name}</span>
                    <div style={{ height: "6px", background: "var(--color-neutral-200)" }}>
                      <div style={{ height: "100%", width: `${r.barPct}%`, background: "var(--color-accent-700)" }}></div>
                    </div>
                    <span style={{ fontSize: "10px", opacity: 0.65, textAlign: "right" }}>{r.count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* DETAILED LOG TABLE (SCROLLABLE INSIDE FRAME) */}
            <div className="card blueprint" style={{ position: "relative", padding: "10px 12px", flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
              <i className="corner tl"></i><i className="corner tr"></i><i className="corner bl"></i><i className="corner br"></i>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px", flexShrink: 0 }}>
                <span className="card-kicker" style={{ fontSize: "9px" }}>NHẬT KÝ CHI TIẾT</span>
                <div style={{ display: "flex", gap: "4px" }}>
                  <button className="btn btn-secondary" style={{ fontSize: "10px", padding: "2px 6px" }} onClick={() => setEntries(loadHistory())}>Làm mới</button>
                  <button className="btn btn-secondary" style={{ fontSize: "10px", padding: "2px 6px", color: "oklch(48% 0.12 35)" }} onClick={() => { clearHistory(); setEntries([]); }}>Xóa log</button>
                </div>
              </div>

              <div style={{ display: "flex", gap: "6px", marginBottom: "8px", flexShrink: 0 }}>
                <input
                  className="input"
                  style={{ flex: 1, height: "26px", fontSize: "11px", padding: "2px 6px" }}
                  placeholder="Lọc từ khóa / khía cạnh…"
                  value={tableSearch}
                  onChange={(e) => setTableSearch(e.target.value)}
                />
                <div className="seg" style={{ display: "flex" }}>
                  <span className={`seg-opt ${tableSentimentFilter === "all" ? "active" : ""}`} style={{ fontSize: "9px", padding: "2px 6px" }} onClick={() => setTableSentimentFilter("all")}>Tất cả</span>
                  <span className={`seg-opt ${tableSentimentFilter === "positive" ? "active" : ""}`} style={{ fontSize: "9px", padding: "2px 6px" }} onClick={() => setTableSentimentFilter("positive")}>Tích cực</span>
                  <span className={`seg-opt ${tableSentimentFilter === "negative" ? "active" : ""}`} style={{ fontSize: "9px", padding: "2px 6px" }} onClick={() => setTableSentimentFilter("negative")}>Tiêu cực</span>
                </div>
              </div>

              {/* Scrollable table container */}
              <div style={{ flex: 1, overflowY: "auto", border: "1px solid var(--color-divider)" }}>
                <table className="table" style={{ fontSize: "11px", margin: 0 }}>
                  <thead>
                    <tr><th style={{ padding: "4px 6px" }}>Giờ</th><th style={{ padding: "4px 6px" }}>Cảm xúc</th><th style={{ padding: "4px 6px" }}>Khía cạnh</th><th style={{ padding: "4px 6px" }}>Nội dung</th></tr>
                  </thead>
                  <tbody>
                    {sessionLog.map((row, idx) => (
                      <tr key={idx}>
                        <td style={{ whiteSpace: "nowrap", opacity: 0.7, padding: "4px 6px" }}>{row.time}</td>
                        <td style={{ padding: "4px 6px" }}><span className="tag" style={{ background: row.sentimentBg, color: row.sentimentColor, fontSize: "9px", padding: "1px 4px" }}>{row.sentimentLabel}</span></td>
                        <td style={{ padding: "4px 6px" }}>
                          <div style={{ display: "flex", gap: "2px", flexWrap: "wrap" }}>
                            {row.negAspects.length > 0 ? (
                              row.negAspects.map((na, i) => (
                                <span key={i} className="tag tag-neutral" style={{ fontSize: "8px", padding: "0 3px" }}>{na}</span>
                              ))
                            ) : (
                              <span style={{ opacity: 0.4 }}>—</span>
                            )}
                          </div>
                        </td>
                        <td style={{ maxWidth: "200px", opacity: 0.8, padding: "4px 6px", fontSize: "10px" }}>{row.snippet}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
