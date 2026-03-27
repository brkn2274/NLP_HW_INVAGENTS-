# Caganbarkin Ustuner — NLP Ödevi
# LLM Tabanlı Çoklu-Ajan İşbirliği Sistemleri

## 🚀 Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Benchmark çalıştır (Bölüm 3 + 4)
python3 -m bolum3_benchmark.benchmark_runner

# 3. PDF rapor oluştur (Bölüm 5)
python3 -m bolum5_rapor.report_generator

# 4. BONUS: Scaling deneyi
python3 scaling_experiment.py

# 5. BONUS: Streamlit dashboard
streamlit run dashboard.py
```

## 📁 Proje Yapısı

```
Caganbarkin_Ustuner_NLP_Odev/
├── README.md
├── requirements.txt
├── dashboard.py                    # BONUS: Streamlit web arayüzü
├── scaling_experiment.py           # BONUS: Ajan scaling deneyi
│
├── bolum1_taksonomi/
│   └── taxonomy.py                 # 4 Enum, Strategy dataclass, S1-S9, classify()
│
├── bolum2_orkestrasyon/
│   ├── agent.py                    # Agent sınıfı (template-based LLM simülasyonu)
│   ├── strategies.py               # S1, S1+, S3, S4, S5, S6 implementasyonları
│   └── orchestrator.py             # Orchestrator sınıfı
│
├── bolum3_benchmark/
│   ├── tasks.json                  # 12 görev (T1×3, T2×3, T3×3, T4×3)
│   └── benchmark_runner.py         # run_benchmark() fonksiyonu
│
├── bolum4_degerlendirme/
│   ├── metrics.py                  # TSR, OQS, CEI, Maliyet
│   └── results/                    # Otomatik oluşturulan çıktılar
│       ├── benchmark_results.csv
│       ├── benchmark_results.json
│       └── evaluation_report.json
│
├── bolum5_rapor/
│   ├── report_generator.py         # PDF + grafik üreticisi
│   ├── rapor.pdf                   # Üretilen PDF rapor
│   ├── fig1_tsr_heatmap.png
│   ├── fig2_token_bar.png
│   ├── fig3_cei_comparison.png
│   └── fig4_cost_vs_perf.png
│
└── logs/                           # Ajan iletişim logları (JSON)
```

## 📊 Bölüm Özeti

### Bölüm 1 — Taksonomi
- `Topology`, `CommunicationProtocol`, `ConflictResolution`, `TaskDecomposition` Enum'ları
- `Strategy` dataclass + S1–S9 stratejileri
- `classify()`: 8 framework destekli (MetaGPT, AutoGen, CrewAI, LangGraph, AutoGPT, BabyAGI, OpenAgents, Camel)

### Bölüm 2 — Orkestrasyon Motoru
- `Agent`: Template-based LLM simülasyonu, rol-farkında yanıt üretimi
- Strateji implementasyonları: **Solo (S1)**, **Solo+Refine (S1+)**, **Sequential Chain (S3)**, **Hierarchical (S4)**, **Debate (S5)**, **Majority Voting (S6)**
- `Orchestrator`: Strateji seçimi, JSON log kayıt

### Bölüm 3 — Benchmark
- 12 görev: T1 Atomik (×3), T2 Bileşik (×3), T3 Çelişkili (×3), T4 Yaratıcı (×3)
- `run_benchmark()`: 72 çalıştırma → CSV + JSON çıktı

### Bölüm 4 — Metrikler
- **TSR**: Task Success Rate
- **OQS**: Output Quality Score (5 kriter × 2 = 10 puan)
- **CEI**: Dengeli, Kalite-Odaklı, Maliyet-Odaklı profil
- **Cost**: GPT-4o fiyatlandırmasıyla maliyet analizi

### Bölüm 5 — Rapor
- 4 grafik: TSR Heatmap, Token Bar, CEI Karşılaştırma, Maliyet Scatter
- PDF rapor: Giriş, Tablolar, Tartışma, Öneriler, Referanslar

## 🎁 Bonus
- ✅ **Scaling Deneyi**: 3, 5, 7 ajan × Majority Voting + Hierarchical
- ✅ **Streamlit Dashboard**: 5 tab — heatmap, token/süre, CEI, maliyet, ham veri

## ⚙️ Sistem Gereksinimleri
- Python 3.10+
- macOS / Linux / Windows

## AI Araç Kullanımı
Bu ödevde Antigravity (Google DeepMind) yapay zeka asistanından destek alınmıştır. Sistem mimarisi, algoritma kararları ve değerlendirme kriterleri, ödev gereksinimlerine uygun olarak benim yönlendirmemle şekillendirilmiştir. Tüm içerik tez önerisindeki kavramlara ve ödev kriterlerine birebir dayanmaktadır.
