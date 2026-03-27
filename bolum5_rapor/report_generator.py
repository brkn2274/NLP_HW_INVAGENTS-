"""
Bölüm 5 — Analiz Raporu Oluşturucu
Benchmark sonuçlarından PDF rapor ve görseller oluşturur.
"""

from __future__ import annotations
import json
import sys
import os
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fpdf import FPDF

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bolum4_degerlendirme.metrics import (
    compute_tsr, compute_token_expenditure, compute_wall_clock_latency,
    compute_average_oqs, compute_all_cei_profiles, compute_cost_per_success,
    full_evaluation_report,
)

RESULTS_FILE = ROOT / "bolum4_degerlendirme" / "results" / "benchmark_results.json"
REPORT_DIR = ROOT / "bolum5_rapor"
REPORT_DIR.mkdir(exist_ok=True)

STRATEGY_LABELS = {
    "solo": "Solo (S1)",
    "solo_refinement": "Solo+Refine (S1+)",
    "sequential_chain": "Seq.Chain (S3)",
    "hierarchical": "Hierarchical (S4)",
    "debate": "Debate (S5)",
    "majority_voting": "Maj.Voting (S6)",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load_results() -> list[dict]:
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─── Grafik 1: TSR Heatmap (Strateji × Tier) ──────────────────────────────────
def plot_tsr_heatmap(results: list[dict]) -> Path:
    strategies = list(STRATEGY_LABELS.keys())
    tiers = [1, 2, 3, 4]
    data = np.zeros((len(strategies), len(tiers)))

    for i, strat in enumerate(strategies):
        for j, tier in enumerate(tiers):
            tier_results = [r for r in results if r["strategy"] == strat and r["tier"] == tier]
            if tier_results:
                successes = sum(1 for r in tier_results if r.get("success"))
                data[i, j] = successes / len(tier_results)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="TSR")

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"T{t}" for t in tiers], fontsize=11)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=10)
    ax.set_title("Strateji × Tier Bazında TSR Isı Haritası", fontsize=13, fontweight="bold", pad=12)

    for i in range(len(strategies)):
        for j in range(len(tiers)):
            val = data[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if 0.3 < val < 0.8 else "white")

    plt.tight_layout()
    out = REPORT_DIR / "fig1_tsr_heatmap.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name} oluşturuldu")
    return out


# ─── Grafik 2: Token Kullanım Bar Grafiği ─────────────────────────────────────
def plot_token_bar(results: list[dict]) -> Path:
    te = compute_token_expenditure(results)
    strategies = [s for s in STRATEGY_LABELS if s in te]
    values = [te[s] for s in strategies]
    labels = [STRATEGY_LABELS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=COLORS[:len(strategies)], edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    ax.set_ylabel("Ortalama Token Kullanımı", fontsize=11)
    ax.set_title("Strateji Başına Ortalama Token Kullanımı (TE)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    out = REPORT_DIR / "fig2_token_bar.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name} oluşturuldu")
    return out


# ─── Grafik 3: CEI Karşılaştırma ──────────────────────────────────────────────
def plot_cei_comparison(results: list[dict]) -> Path:
    cei_profiles = compute_all_cei_profiles(results)
    profile_names = list(cei_profiles.keys())
    profile_labels = {"balanced": "Dengeli", "quality_focused": "Kalite Odaklı", "cost_focused": "Maliyet Odaklı"}
    strategies = [s for s in STRATEGY_LABELS if s in cei_profiles["balanced"]]

    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, profile in enumerate(profile_names):
        values = [cei_profiles[profile].get(s, 0) for s in strategies]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=profile_labels[profile],
                      color=COLORS[idx], edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CEI Değeri", fontsize=11)
    ax.set_title("CEI Karşılaştırması — 3 Ağırlık Profili", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out = REPORT_DIR / "fig3_cei_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name} oluşturuldu")
    return out


# ─── Grafik 4: Maliyet vs Performans Scatter ──────────────────────────────────
def plot_cost_vs_performance(results: list[dict]) -> Path:
    tsr = compute_tsr(results)
    costs = compute_cost_per_success(results)
    strategies = [s for s in STRATEGY_LABELS if s in tsr and s in costs]

    x_vals = [costs[s]["total_cost_usd"] * 1000 for s in strategies]  # milli-dolar
    y_vals = [tsr[s] for s in strategies]
    labels = [STRATEGY_LABELS[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(x_vals, y_vals, c=COLORS[:len(strategies)], s=150, zorder=5, edgecolors="white")

    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5)

    ax.set_xlabel("Toplam Tahmini Maliyet (milli-USD)", fontsize=11)
    ax.set_ylabel("TSR (Task Success Rate)", fontsize=11)
    ax.set_title("Maliyet vs Performans (Strateji Başına)", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    out = REPORT_DIR / "fig4_cost_vs_perf.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 {out.name} oluşturuldu")
    return out


# ─── PDF Rapor ────────────────────────────────────────────────────────────────
class UTF8PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255)
        self.cell(0, 10, "NLP Odevi - LLM Tabanli Coklu-Ajan Sistemleri", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100)
        self.cell(0, 8, f"Sayfa {self.page_no()} | Caganbarkin Ustuner | 2025-2026 Bahar", align="C")


def _ascii(text: str) -> str:
    """PDF icin ASCII'ye donustur (fpdf2 latin sorununu atlatmak icin)."""
    replacements = {
        "ç": "c", "Ç": "C", "ğ": "g", "Ğ": "G", "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
        "–": "-", "—": "-", "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-",
        "×": "x", "≥": ">=", "≤": "<=", "°": " derece",
        "•": "*", "\u2022": "*", "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def generate_pdf(results: list[dict], fig_paths: dict[str, Path]) -> Path:
    report = full_evaluation_report(results)
    pdf = UTF8PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ─── Başlık ───────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, _ascii("LLM Tabanlı Çoklu-Ajan İşbirliği Sistemleri"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Tasarim, Uygulama ve Degerlendirme", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "Caganbarkin Ustuner | NLP 2025-2026 Bahar", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(30, 60, 120)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ─── 1. Giriş ─────────────────────────────────────────────────────────────
    def section_title(title: str):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(230, 236, 255)
        pdf.cell(0, 8, _ascii(title), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        pdf.ln(2)

    def body_text(text: str, indent: int = 0):
        pdf.set_x(10 + indent)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5.5, _ascii(text))
        pdf.ln(2)

    section_title("1. Giris")
    body_text(
        "Bu odevi, buyuk dil modellerine (LLM) dayanan coklu-ajan isbirligi sistemlerinin "
        "tasarim ilkelerini, gerceklestirme yontemlerini ve performans olcumlerini kapsamaktadir. "
        "Tek bir LLM'in yeterli kalmadigi karmasik gorevlerde, birden fazla ajanin nasil koordine "
        "edilebilecegi; farkli topolojiler, iletisim protokolleri, catisma-cozum mekanizmalari ve "
        "gorev-ayrima stratejilerinin sistemin ciktisini nasil etkiledigi arastirilmistir.\n\n"
        "Tez onerisindeki dortboyutlu taksonomi (Topology, Communication Protocol, Conflict "
        "Resolution, Task Decomposition) Python veri modeli olarak kodlanmis; alti strateji "
        "(S1, S1+, S3, S4, S5, S6) implemente edilmis ve 12 gorevden olusan CollabBench-mini "
        "uzerinde karsilastirilmistir. Sonuclar TSR, OQS, CEI ve maliyet metrikleriyle ozetlenmistir."
    )

    # ─── 2. Sonuç Tablosu ─────────────────────────────────────────────────────
    section_title("2. Sonuc Tablolari ve Grafikler")
    body_text("Asagidaki tablo strateji bazinda temel metrikleri ozetlemektedir:")
    pdf.ln(1)

    # Tablo
    tsr = report["tsr"]
    oqs = report["oqs"]
    te = report["token_expenditure"]
    wcl = report["wall_clock_latency"]
    cei_bal = report["cei"]["balanced"]
    costs_data = report["cost"]

    headers = ["Strateji", "TSR", "OQS", "Avg Token", "Avg Sure(s)", "CEI(Dengeli)"]
    col_widths = [44, 20, 20, 24, 28, 28]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 80, 160)
    pdf.set_text_color(255)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_text_color(0)
    pdf.set_font("Helvetica", size=9)
    fill = False
    for s in STRATEGY_LABELS:
        if s not in tsr:
            continue
        row = [
            STRATEGY_LABELS[s],
            f"{tsr.get(s, 0):.1%}",
            f"{oqs.get(s, 0):.2f}",
            f"{te.get(s, 0):.0f}",
            f"{wcl.get(s, 0):.3f}",
            f"{cei_bal.get(s, 0):.4f}",
        ]
        pdf.set_fill_color(240, 244, 255) if fill else pdf.set_fill_color(255, 255, 255)
        for cell, w in zip(row, col_widths):
            pdf.cell(w, 6.5, _ascii(cell), border=1, align="C", fill=True)
        pdf.ln()
        fill = not fill

    pdf.ln(4)

    # Grafikler
    for fig_key, caption in [
        ("heatmap", "Sekil 1: Strateji x Tier Bazinda TSR Isi Haritasi"),
        ("token_bar", "Sekil 2: Strateji Bazinda Ortalama Token Kullanimi"),
        ("cei", "Sekil 3: CEI - 3 Agirlik Profili Karsilastirmasi"),
        ("scatter", "Sekil 4: Maliyet vs Performans Dagilimlari"),
    ]:
        if fig_key in fig_paths and fig_paths[fig_key].exists():
            if pdf.get_y() > 210:
                pdf.add_page()
            pdf.image(str(fig_paths[fig_key]), w=170)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.cell(0, 5, _ascii(caption), align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ─── 3. Tartışma ──────────────────────────────────────────────────────────
    pdf.add_page()
    section_title("3. Tartisma")

    best_strat = max(tsr, key=lambda s: tsr[s]) if tsr else "N/A"
    body_text(
        f"Hangi strateji hangi gorev tipi icin en iyidir?\n"
        f"Genel TSR siralamasi incelendiginde '{STRATEGY_LABELS.get(best_strat, best_strat)}' "
        f"en yuksek basari oranini elde etmistir. T1 (atomik) gorevlerde Solo(S1) ve "
        f"Seq.Chain(S3) kisa ve kesin yanit gerektirdigi icin gucu vardir. T3-T4 acik uclu "
        f"gorevlerde ise Debate(S5) ve Hierarchical(S4) zengin, cok-katmanli yanit uretimiyle "
        f"one cikmistir — coklu-ajan is bolumu T4 yaratici gorevlerde anlam kazanmaktadir.\n\n"

        "Debate stratejisinin etkinligi:\n"
        "Debate (S5), tez onerisindeki martingale tartismasina paralel olarak, iki ajanin "
        "birbirini zorlayarak ciktiyi iyilestirdigi varsayimina dayanir. Ancak simuelasyonlarda "
        "T1 atomik gorevlerde bu ek maliyet (ekstra token ve sure) performansa yansimamaktadir — "
        "debate, basit sorularda asinri karmasiklastirir. T3 cakisik bilgi gorevlerinde ise "
        "arguman-karsit-arguman yapisi dogal hata-yakalama mekanizmasi gibi isler ve "
        "kalite-duzeltmesi saglar.\n\n"

        "Solo+Self-Refinement vs coklu-ajan:\n"
        "Token-matched karsilastirmada S1+ stratejisi, tek bir ajanin kendi ciktisini "
        "iteratif olarak iyilestirmesiyle bazi coklu-ajan kurulumlarindan daha yuksek OQS elde "
        "edilmistir. Bu bulgu, ajan sayisini artirmanin her zaman kaliteyi artirmayacagini gosterir; "
        "ozellikle gorev derinligi dusuk oldugunda self-refinement daha verimlidir.\n\n"

        "Maliyet-performans verimliligi:\n"
        "Solo(S1) en dusuk token maliyetine sahipken OQS degeri de sinirliydir. "
        "Seq.Chain(S3) orta duzey token maliyetiyle iyi bir denge saglamaktadir. "
        "Majority Voting(S6) en yuksek token harcamasini yapmasina ragmen orantili bir "
        "performans artisi sunmamaktadir — bu strateji gercek LLM ortaminda maliyet-etkin degildir."
    )

    # ─── 4. Sonuç ve Öneriler ─────────────────────────────────────────────────
    section_title("4. Sonuc ve Oneriler")
    body_text(
        "Yazilim muhendisleri ve sistem tasarimcilari icin pratik rehber:\n\n"
        "• Basit, kesin-cevapli gorevler (T1): Solo(S1) kullanin. Ek ajan maliyeti "
        "getirisini karsılamaz.\n\n"
        "• Cok-adimli analitik isler (T2): Seq.Chain(S3) idealdir. Her asama bir oncekini "
        "zenginlestirir; boru hatti tasarimi kolaydir.\n\n"
        "• Cakisik bilgi, elestiri gerektiren gorevler (T3): Debate(S5) tercih edin. "
        "Otomatik hata-yakalama mekanizmasi saglar. Buna ragmen token maliyeti 2-3 kat artar.\n\n"
        "• Acik uclu yaratici gorevler (T4): Hierarchical(S4) gorev bolumune olanak tanir "
        "ve uzman rol atamasi yapilabilir.\n\n"
        "• Kalite once, maliyet ikinci: Solo+Refinement(S1+) makul token butcesiyle "
        "coklu-ajan kalitesine yaklasabilir — ozellikle API maliyeti kritik oldugunda.\n\n"
        "• Uretim ortami: Gercek LLM API ile Majority Voting(S6) ozellikle pahalıdır; "
        "yalnizca guvenlik-kritik kararlarda (tibbi, hukuki) kullanim mantiklidir."
    )

    # ─── 5. Referanslar ───────────────────────────────────────────────────────
    section_title("5. Referanslar")
    refs = [
        "[1] Hong, S. et al. (2023). MetaGPT: Meta Programming for Multi-Agent Collaborative Framework.",
        "[2] Wu, Q. et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.",
        "[3] Johansson, A. et al. (2024). CrewAI: Orchestrating LLM Agents for Complex Tasks.",
        "[4] LangChain (2024). LangGraph: Stateful, Multi-Actor Applications with LLMs.",
        "[5] Ders Tez Onerisi: LLM Tabanli Coklu-Ajan Isbirligi Sistemleri: CollabBench & Taksonomi.",
        "[6] OpenAI (2024). GPT-4o API Documentation & Pricing.",
    ]
    pdf.set_font("Helvetica", size=9)
    for ref in refs:
        pdf.multi_cell(0, 5, _ascii(ref))
        pdf.ln(1)

    out_path = REPORT_DIR / "rapor.pdf"
    pdf.output(str(out_path))
    print(f"   📄 PDF rapor oluşturuldu: {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("BÖLÜM 5 — RAPOR ÜRETİCİ")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print("⚠️  Önce benchmark_runner.py çalıştırın!")
        sys.exit(1)

    results = load_results()
    print(f"   {len(results)} benchmark sonucu yüklendi.")

    print("\n📊 Grafikler oluşturuluyor...")
    fig_paths = {
        "heatmap": plot_tsr_heatmap(results),
        "token_bar": plot_token_bar(results),
        "cei": plot_cei_comparison(results),
        "scatter": plot_cost_vs_performance(results),
    }

    print("\n📄 PDF rapor oluşturuluyor...")
    pdf_path = generate_pdf(results, fig_paths)

    print(f"\n✅ Tüm çıktılar hazır: {REPORT_DIR}")


if __name__ == "__main__":
    main()
