// Mirrors the Pydantic schemas exposed by the FastAPI gateway.

export type Sentiment = "negative" | "neutral" | "positive";

export interface AbsaAspect {
  aspect: string;
  label: string;
  sentiment: Sentiment;
  presence: number;
}

export interface AbsaResult {
  text: string;
  overall: Sentiment;
  overall_probs: number[];
  aspects: AbsaAspect[];
}

export interface TikiSample {
  original_text: string;
  text: string;
  order_code: string;
  product_id: number;
  stars: number;
}

export interface TeamMember {
  id: number;
  email: string;
  name: string;
  role: string;
  is_active: boolean;
  shop_id: number;
  created_at: string;
}

export interface HistorySummaryBucket {
  period: string;
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  aspect_negative: Record<string, number>;
}

export interface HistorySummary {
  groupby: "week" | "month";
  buckets: HistorySummaryBucket[];
}
