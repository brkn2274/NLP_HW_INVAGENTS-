"""
BONUS — Streamlit Dashboard (Premium Edition)
Benchmark sonuçlarını interaktif olarak gösteren premium web arayüzü.
Çalıştır: streamlit run dashboard.py
"""

import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).parent
RESULTS_FILE = ROOT / "bolum4_degerlendirme" / "results" / "benchmark_results.json"
REPORT_DIR = ROOT / "bolum5_rapor"

sys.path.insert(0, str(ROOT))
from bolum4_degerlendirme.metrics import (
    compute_tsr, compute_token_expenditure, compute_wall_clock_latency,
    compute_all_cei_profiles, compute_cost_per_success,
)

STRATEGY_LABELS = {
    "solo": "Solo (S1)",
    "solo_refinement": "Solo+Refine (S1+)",
    "sequential_chain": "Seq.Chain (S3)",
    "hierarchical": "Hierarchical (S4)",
    "debate": "Debate (S5)",
    "majority_voting": "Maj.Voting (S6)",
}

# Premium Dark Mode Renk Paleti (Neon / Pastel karışımı)
COLORS = ["#38bdf8", "#fb923c", "#34d399", "#f87171", "#a78bfa", "#fbcfe8"]

def apply_custom_css():
    st.markdown("""
        <style>
            /* Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Outfit', sans-serif;
            }

            /* Ana Arka Planı Koyu Temaya Çekme */
            .stApp {
                background-color: #0f172a;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(30, 27, 75, 1) 0, transparent 50%), 
                    radial-gradient(at 100% 0%, rgba(15, 23, 42, 1) 0, transparent 50%);
                color: #f8fafc;
            }

            /* Premium Header */
            .header-container {
                background: linear-gradient(135deg, rgba(30, 58, 138, 0.6) 0%, rgba(88, 28, 135, 0.6) 100%);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 3rem 2rem;
                text-align: center;
                margin-bottom: 2.5rem;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .header-container::before {
                content: '';
                position: absolute;
                top: -50%; left: -50%; width: 200%; height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
                pointer-events: none;
            }

            .header-title {
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(to right, #e0e7ff, #38bdf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -1px;
            }

            .header-subtitle {
                font-size: 1.2rem;
                color: #94a3b8;
                margin-top: 0.8rem;
                font-weight: 300;
                letter-spacing: 2px;
                text-transform: uppercase;
            }

            /* Metrik Kartı Yapısı (Glassmorphism) */
            .metric-card {
                background: rgba(30, 41, 59, 0.5);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 1.5rem;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .metric-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 45px rgba(56, 189, 248, 0.15);
                border-color: rgba(56, 189, 248, 0.3);
                background: rgba(30, 41, 59, 0.8);
            }

            .metric-title {
                font-size: 0.9rem;
                color: #cbd5e1;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-weight: 600;
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 800;
                margin: 0;
                background: linear-gradient(135deg, #fff, #94a3b8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.2;
            }
            
            .metric-value.highlight {
                background: linear-gradient(135deg, #38bdf8, #818cf8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .metric-desc {
                font-size: 0.85rem;
                color: #64748b;
                margin-top: 0.4rem;
                font-weight: 400;
            }

            /* Tabların tasarımı */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: rgba(15, 23, 42, 0.5);
                padding: 8px;
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            .stTabs [data-baseweb="tab"] {
                height: 40px;
                white-space: pre-wrap;
                background-color: transparent;
                border-radius: 12px;
                color: #94a3b8;
                padding: 0 16px;
                font-weight: 600;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
                color: white !important;
                border: none !important;
            }
            
            /* Dataframe ve Expander Koyu Teması */
            [data-testid="stDataFrame"] {
                background: rgba(30,41,59,0.5);
                border-radius: 16px;
                padding: 10px;
                border: 1px solid rgba(255,255,255,0.05);
            }
        </style>
    """, unsafe_allow_html=True)


def configure_matplotlib_dark_theme():
    """Matplotlib grafiklerini sayfaya uygun premium karanlık temaya çevirir."""
    plt.style.use('dark_background')
    matplotlib.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none',
        'axes.edgecolor': '#334155',
        'axes.labelcolor': '#cbd5e1',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'text.color': '#f8fafc',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#334155',
        'grid.alpha': 0.3,
        'font.family': 'sans-serif'
    })


@st.cache_data
def load_data():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def render_metric_card(icon: str, title: str, value: str, desc: str, highlight: bool = False):
    """HTML+CSS ile özel metrik kartı çizer."""
    highlight_class = "highlight" if highlight else ""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{icon} {title}</div>
        <div class="metric-value {highlight_class}">{value}</div>
        <div class="metric-desc">{desc}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="CollabBench | Multi-Agent AI",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_css()
    configure_matplotlib_dark_theme()

    # ─── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">⚛️ CollabBench Multi-Agent Motoru</h1>
            <p class="header-subtitle">LLM İşbirliği Stratejileri • Taksonomi Analizi • Performans Metrikleri</p>
        </div>
    """, unsafe_allow_html=True)

    if not RESULTS_FILE.exists():
        st.error("⚠️ Benchmark sonuçları bulunamadı. Lütfen önce testleri çalıştırın.")
        return

    results = load_data()
    df = pd.DataFrame(results)

    # ─── Sidebar Tasarımı ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h2 style='color:#38bdf8; font-weight:800;'>🎛️ KONTROL PANELİ</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:2rem;'>Filtreleri kullanarak analiz görünümünü özelleştirin.</p>", unsafe_allow_html=True)
        
        selected_strategies = st.multiselect(
            "Hedef Stratejiler",
            options=list(STRATEGY_LABELS.keys()),
            default=list(STRATEGY_LABELS.keys()),
            format_func=lambda x: STRATEGY_LABELS[x],
        )
        selected_tiers = st.multiselect(
            "Görev Zorluk Seviyeleri",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"T{x} — {'Atomik' if x==1 else 'Bileşik' if x==2 else 'Çelişkili' if x==3 else 'Yaratıcı'}",
        )
        cei_profile = st.selectbox(
            "CEI Ağırlık Profili",
            options=["balanced", "quality_focused", "cost_focused"],
            format_func=lambda x: {"balanced": "⚖️ Dengeli", "quality_focused": "🎯 Kalite Odaklı (TSR & OQS)", "cost_focused": "💰 Maliyet Odaklı (Token & Süre)"}[x],
        )
        
        st.markdown("---")
        st.markdown("<div style='text-align:center; font-size:0.8rem; color:#64748b;'>NLP Projesi<br>Çağanbarkın Üstüner</div>", unsafe_allow_html=True)

    filtered_df = df[df["strategy"].isin(selected_strategies) & df["tier"].isin(selected_tiers)]
    filtered_results = filtered_df.to_dict("records")

    # ─── Premium Metrik Kartları ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    tsr_all = compute_tsr(filtered_results)
    best_strat = max(tsr_all, key=tsr_all.get) if tsr_all else "-"
    avg_tsr = sum(tsr_all.values()) / len(tsr_all) if tsr_all else 0
    total_tokens = sum(r["total_tokens"] for r in filtered_results)
    success_count = sum(1 for r in filtered_results if r.get("success"))

    with col1:
        render_metric_card("🏆", "EN İYİ STRATEJİ", f"{tsr_all.get(best_strat, 0):.0%}", f"{STRATEGY_LABELS.get(best_strat, best_strat)}", highlight=True)
    with col2:
        render_metric_card("📈", "ORTALAMA BAŞARI", f"{avg_tsr:.1%}", f"{success_count} / {len(filtered_results)} Toplam Görev")
    with col3:
        render_metric_card("⚡", "TOPLAM İŞLEM", str(len(filtered_results)), f"{len(selected_strategies)} Strateji × {len(selected_tiers)} Tier")
    with col4:
        render_metric_card("🧠", "AĞ YÜKÜ (TOKEN)", f"{total_tokens:,}", "Tüm süreçte harcanan token")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ─── Modüler Tab Düzeni ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 TSR Heatmap", 
        "📊 Hız & Kaynak Analizi", 
        "⚖️ CEI Karşılaştırması", 
        "💎 Maliyet Optimizasyonu", 
        "📂 Ham Terminal Verisi"
    ])

    # 🔥 Tab 1: TSR Heatmap
    with tab1:
        st.markdown("<h3 style='color:#e2e8f0;'>Strateji × Zorluk: Başarı Dağılımı</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Kırmızıdan yeşile doğru başarı oranının artışını gösteren strateji yorgunluk/kapasite matrisi.</p><br>", unsafe_allow_html=True)
        
        strategies = [s for s in STRATEGY_LABELS if s in selected_strategies]
        tiers = [t for t in [1, 2, 3, 4] if t in selected_tiers]
        
        if len(strategies) > 0 and len(tiers) > 0:
            data = np.zeros((len(strategies), len(tiers)))
            for i, strat in enumerate(strategies):
                for j, tier in enumerate(tiers):
                    tier_res = [r for r in filtered_results if r["strategy"] == strat and r["tier"] == tier]
                    if tier_res:
                        data[i, j] = sum(1 for r in tier_res if r.get("success")) / len(tier_res)

            fig, ax = plt.subplots(figsize=(10, max(4, len(strategies)*0.8)))
            
            # Premium Colormap
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", ["#ef4444", "#f59e0b", "#10b981"])
            
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation='nearest')
            
            # Kenarlıkları Yumuşatma & Grid İptali
            for spine in ax.spines.values():
                spine.set_color('#1e293b')
                spine.set_linewidth(2)
            ax.grid(False)

            ax.set_xticks(range(len(tiers)))
            ax.set_xticklabels([f"Tier {t}" for t in tiers], fontsize=12, fontweight='bold', color="#cbd5e1")
            ax.set_yticks(range(len(strategies)))
            ax.set_yticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=12, fontweight='bold', color="#cbd5e1")
            
            # Değerleri ortala ve estetik yazdır
            for i in range(len(strategies)):
                for j in range(len(tiers)):
                    v = data[i, j]
                    text_color = "white" if (v < 0.4 or v > 0.7) else "#1e293b"
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=14, fontweight="900", color=text_color)
            
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()

    # 📊 Tab 2: Token & Süre
    with tab2:
        c1, c2 = st.columns(2)
        te = compute_token_expenditure(filtered_results)
        wcl = compute_wall_clock_latency(filtered_results)
        active_strategies = [s for s in STRATEGY_LABELS if s in te]

        with c1:
            st.markdown("<h4 style='color:#38bdf8;'>Ağ Veri Yükü (Token Genişliği)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            vals = [te[s] for s in active_strategies]
            bars = ax.bar([STRATEGY_LABELS[s] for s in active_strategies], vals,
                          color=COLORS[:len(active_strategies)], edgecolor="none", alpha=0.9, width=0.6)
            
            ax.bar_label(bars, fmt="%.0f", padding=6, fontsize=11, fontweight="bold", color="white")
            ax.set_ylabel("Ortalama Token / Görev", fontsize=11, fontweight="bold")
            plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()

        with c2:
            st.markdown("<h4 style='color:#a78bfa;'>Yanıt Gecikmesi (Latency)</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            vals = [wcl[s] for s in active_strategies]
            bars = ax.bar([STRATEGY_LABELS[s] for s in active_strategies], vals,
                          color=COLORS[::-1][:len(active_strategies)], edgecolor="none", alpha=0.9, width=0.6)
            
            ax.bar_label(bars, fmt="%.3fs", padding=6, fontsize=11, fontweight="bold", color="white")
            ax.set_ylabel("Süre (Saniye)", fontsize=11, fontweight="bold")
            plt.xticks(rotation=30, ha="right", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()

    # ⚖️ Tab 3: CEI
    with tab3:
        st.markdown(f"<h3 style='color:#e2e8f0;'>İşbirliği Verimlilik Endeksi (CEI)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#94a3b8; font-size:0.9rem;'>Seçili Profil: <b>{cei_profile.replace('_', ' ').upper()}</b> — Stratejilerin maliyet/başarı verimliliğini puanlar.</p><br>", unsafe_allow_html=True)
        
        cei_profiles = compute_all_cei_profiles(filtered_results)
        active_strategies = [s for s in STRATEGY_LABELS if s in cei_profiles.get(cei_profile, {})]

        x = np.arange(len(active_strategies))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        
        profile_data = {
            "balanced": (cei_profiles.get("balanced", {}), "Dengeli", "#38bdf8"),
            "quality_focused": (cei_profiles.get("quality_focused", {}), "Kalite Odaklı", "#10b981"),
            "cost_focused": (cei_profiles.get("cost_focused", {}), "Maliyet Odaklı", "#f59e0b"),
        }
        
        for idx, (pkey, (pdata, plabel, pcolor)) in enumerate(profile_data.items()):
            vals = [pdata.get(s, 0) for s in active_strategies]
            
            # Vurgulama
            alpha = 1.0 if pkey == cei_profile else 0.3
            ax.bar(x + (idx - 1) * width, vals, width, label=plabel, color=pcolor, alpha=alpha, edgecolor="none", zorder=3)
            
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in active_strategies], rotation=20, ha="right", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalize CEI Değeri", fontsize=11, fontweight="bold")
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        ax.axhline(0, color="#475569", linewidth=2, linestyle="-", zorder=1)
        plt.tight_layout()
        st.pyplot(fig, transparent=True)
        plt.close()

    # 💎 Tab 4: Maliyet
    with tab4:
        st.markdown("<h3 style='color:#e2e8f0;'>Maliyet / Performans Korrelasyonu</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>GPT-4o API referans alınarak simüle edilmiş finansal verimlilik. Sağ üst köşe ideal noktayı (Yüksek TSR, Düşük Maliyet) belirtmez. <b>Sol üst köşe idealdir.</b></p><br>", unsafe_allow_html=True)
        
        costs = compute_cost_per_success(filtered_results)
        tsr_d = compute_tsr(filtered_results)
        active_strategies2 = [s for s in STRATEGY_LABELS if s in costs and s in tsr_d]
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, s in enumerate(active_strategies2):
                x_val = costs[s]["total_cost_usd"] * 1000 # milli-usd
                y_val = tsr_d[s]
                
                scatter = ax.scatter(x_val, y_val, color=COLORS[i % len(COLORS)], s=300, zorder=5, alpha=0.8, edgecolor="#f8fafc", linewidth=2)
                
                ax.annotate(STRATEGY_LABELS[s], (x_val, y_val), xytext=(12, 0), textcoords="offset points", 
                            fontsize=11, fontweight="bold", color="#cbd5e1", va="center")
            
            ax.set_xlabel("Toplam Maliyet (milli-USD)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Task Success Rate (TSR)", fontsize=12, fontweight="bold")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            plt.tight_layout()
            st.pyplot(fig, transparent=True)
            plt.close()
            
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            cost_rows = []
            for s in active_strategies2:
                c = costs[s]
                cost_rows.append({
                    "Strateji / Ajan Modeli": STRATEGY_LABELS[s],
                    "Maliyet ($)": f"${c['total_cost_usd']:.5f}",
                    "Cost/Başarı": f"${c['cost_per_success_usd']:.5f}",
                })
            styled_df = pd.DataFrame(cost_rows).set_index("Strateji / Ajan Modeli")
            st.dataframe(styled_df, use_container_width=True, height=400)

    # 📂 Tab 5: Ham Veri
    with tab5:
        st.markdown("<h3 style='color:#e2e8f0;'>Detaylı JSON Veri Tablosu</h3>", unsafe_allow_html=True)
        display_df = filtered_df.copy()
        display_df["strategy"] = display_df["strategy"].map(STRATEGY_LABELS).fillna(display_df["strategy"])
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ VERİYİ CSV OLARAK DIŞA AKTAR",
            data=filtered_df.to_csv(index=False),
            file_name="collabbench_results.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
