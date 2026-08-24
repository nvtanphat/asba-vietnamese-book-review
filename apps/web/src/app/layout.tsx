import type { Metadata } from "next";
import { Source_Serif_4, Be_Vietnam_Pro, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/features/app-shell";

const displaySerif = Source_Serif_4({
  variable: "--font-display",
  subsets: ["vietnamese", "latin"],
  weight: ["500", "600", "700"],
});
const bodySans = Be_Vietnam_Pro({
  variable: "--font-body",
  subsets: ["vietnamese", "latin"],
  weight: ["400", "500", "600", "700"],
});
const utilityMono = IBM_Plex_Mono({
  variable: "--font-utility",
  subsets: ["vietnamese", "latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "SentenAI — ABSA Predictor Dashboard",
  description:
    "Trực quan hóa dự đoán sắc thái tổng quát và 6 khía cạnh đánh giá từ mô hình PhoBERT ABSA.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="vi"
      className={`${displaySerif.variable} ${bodySans.variable} ${utilityMono.variable} h-full antialiased`}
    >
      <head>
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
        />
      </head>
      <body className="min-h-full font-body transition-colors duration-150">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
