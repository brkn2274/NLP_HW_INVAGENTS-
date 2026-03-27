"""
Bölüm 4 — Değerlendirme ve Metrikler
TSR, OQS, WCL, TE, CEI, Maliyet hesaplamaları
"""

from __future__ import annotations
import json
import re
import math
from pathlib import Path
from typing import Any, Optional

from bolum2_orkestrasyon.strategies import OrchestratorResult


# ─── Token Fiyatlandırması ────────────────────────────────────────────────────
# GPT-4o fiyatları (Mart 2026 referansı)
PRICE_PER_INPUT_TOKEN = 2.50 / 1_000_000   # $2.50 / 1M input token
PRICE_PER_OUTPUT_TOKEN = 10.00 / 1_000_000  # $10.00 / 1M output token


# ─── Yardımcı: Basit string benzerliği ───────────────────────────────────────
def _text_similarity(pred: str, expected: str) -> float:
    """
    Basit token-jaccard benzerliği.
    Gerçek NLP'de BERTScore veya ROUGE kullanılır.
    """
    pred_tokens = set(pred.lower().split())
    exp_tokens = set(expected.lower().split())
    if not exp_tokens:
        return 0.0
    intersection = pred_tokens & exp_tokens
    union = pred_tokens | exp_tokens
    return len(intersection) / len(union) if union else 0.0


def _check_exact_or_fuzzy_match(prediction: str, expected: str, threshold: float = 0.3) -> bool:
    """Tam eşleşme veya bulanık eşleşme kontrolü."""
    pred_clean = prediction.lower().strip()
    exp_clean = expected.lower().strip()

    # Tam eşleşme
    if exp_clean in pred_clean:
        return True

    # Sayısal eşleşme (T1 matematik soruları için)
    numbers_in_pred = re.findall(r'\d+(?:[.,]\d+)?', pred_clean)
    numbers_in_exp = re.findall(r'\d+(?:[.,]\d+)?', exp_clean)
    if numbers_in_exp and any(n in numbers_in_pred for n in numbers_in_exp):
        return True

    # Anahtar kelime eşleşmesi
    similarity = _text_similarity(pred_clean, exp_clean)
    return similarity >= threshold


# ─── OQS (Output Quality Score) ───────────────────────────────────────────────
def compute_oqs(
    prediction: str,
    task_prompt: str,
    expected_answer: Optional[str] = None,
    rubric: Optional[dict[str, int]] = None,
    tier: int = 1,
) -> float:
    """
    Output Quality Score hesaplar (1-10 arası).

    Kriterler (her biri 0-2 puan):
    1. Doğruluk / İlgililik
    2. Bütünlük / Tamlık
    3. Tutarlılık
    4. Derinlik / Yaratıcılık
    5. Kısıt Uyumu
    """
    scores = {}

    pred_words = prediction.split()
    pred_len = len(pred_words)
    task_words = set(task_prompt.lower().split())
    pred_words_set = set(prediction.lower().split())

    # 1. Doğruluk / İlgililik (0-2)
    if expected_answer:
        if _check_exact_or_fuzzy_match(prediction, expected_answer):
            scores["accuracy"] = 2.0
        else:
            sim = _text_similarity(prediction, expected_answer)
            scores["accuracy"] = min(2.0, sim * 4)
    else:
        # Görevle örtüşme
        overlap = len(task_words & pred_words_set) / max(len(task_words), 1)
        scores["accuracy"] = min(2.0, overlap * 4)

    # 2. Bütünlük / Tamlık (0-2)
    # Uzunluk bazlı yaklaşık değerlendirme
    tier_min_words = {1: 10, 2: 30, 3: 50, 4: 80}
    min_words = tier_min_words.get(tier, 20)
    if pred_len >= min_words * 2:
        scores["completeness"] = 2.0
    elif pred_len >= min_words:
        scores["completeness"] = 1.5
    elif pred_len >= min_words // 2:
        scores["completeness"] = 1.0
    else:
        scores["completeness"] = 0.5

    # 3. Tutarlılık (0-2)
    # Çelişkili kelime çiftleri kontrolü (basit sezgisel)
    contradiction_patterns = [
        ("evet", "hayır"), ("doğru", "yanlış"), ("artıyor", "azalıyor"),
        ("yes", "no"), ("true", "false"),
    ]
    has_contradiction = any(
        p1 in prediction.lower() and p2 in prediction.lower()
        for p1, p2 in contradiction_patterns
    )
    scores["consistency"] = 1.0 if has_contradiction else 2.0

    # 4. Derinlik / Yaratıcılık (0-2)
    # Benzersiz kelime oranı × uzunluk faktörü
    unique_ratio = len(set(pred_words)) / max(pred_len, 1)
    depth_score = min(2.0, unique_ratio * 3 + (0.5 if pred_len > 50 else 0))
    scores["depth"] = depth_score

    # 5. Kısıt Uyumu (0-2)
    # Göreve özgü kısıtlar: kodlama görevi → kod bloğu var mı?
    constraint_score = 2.0
    if "python" in task_prompt.lower() or "fonksiyon" in task_prompt.lower():
        if "def " not in prediction and "```" not in prediction:
            constraint_score = 1.0
    if "200 kelime" in task_prompt.lower():
        if abs(pred_len - 200) > 100:
            constraint_score = 1.0
    scores["constraint"] = constraint_score

    # Toplam (5 kriter × 2 = 10 max)
    total = sum(scores.values())
    return round(min(10.0, max(1.0, total)), 2)


# ─── Görev Başarısı Değerlendirmesi ──────────────────────────────────────────
def evaluate_result(
    result: OrchestratorResult,
    expected_answer: Optional[str],
    rubric: Optional[dict],
    tier: int,
) -> dict[str, Any]:
    """
    Tek bir çalıştırmanın metriklerini hesaplar.

    Returns:
        dict içinde: success, tsr, oqs
    """
    prediction = result.final_answer

    # T1-T2: Expected answer varsa match kontrolü
    if expected_answer is not None:
        success = _check_exact_or_fuzzy_match(prediction, expected_answer)
    else:
        # T3-T4: OQS ≥ 6 ise başarılı say
        oqs_val = compute_oqs(
            prediction=prediction,
            task_prompt=result.task,
            expected_answer=None,
            rubric=rubric,
            tier=tier,
        )
        success = oqs_val >= 6.0

    oqs = compute_oqs(
        prediction=prediction,
        task_prompt=result.task,
        expected_answer=expected_answer,
        rubric=rubric,
        tier=tier,
    )

    return {
        "success": success,
        "tsr": 1.0 if success else 0.0,
        "oqs": oqs,
    }


# ─── Toplu Metrik Hesaplamaları ───────────────────────────────────────────────
def compute_tsr(results: list[dict]) -> dict[str, float]:
    """
    Task Success Rate per strategy.
    TSR = başarılı görev sayısı / toplam görev sayısı
    """
    from collections import defaultdict
    strategy_data: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        s = r["strategy"]
        strategy_data[s]["total"] += 1
        strategy_data[s]["success"] += int(r.get("success", False))

    return {
        s: d["success"] / d["total"] if d["total"] > 0 else 0.0
        for s, d in strategy_data.items()
    }


def compute_token_expenditure(results: list[dict]) -> dict[str, float]:
    """
    Token Expenditure (TE): Strateji başına ortalama token kullanımı.
    """
    from collections import defaultdict
    strategy_data: dict[str, list] = defaultdict(list)
    for r in results:
        strategy_data[r["strategy"]].append(r.get("total_tokens", 0))
    return {
        s: sum(tokens) / len(tokens) if tokens else 0.0
        for s, tokens in strategy_data.items()
    }


def compute_wall_clock_latency(results: list[dict]) -> dict[str, float]:
    """
    Wall-Clock Latency (WCL): Strateji başına ortalama çalışma süresi (saniye).
    """
    from collections import defaultdict
    strategy_data: dict[str, list] = defaultdict(list)
    for r in results:
        strategy_data[r["strategy"]].append(r.get("elapsed_time_sec", 0.0))
    return {
        s: sum(times) / len(times) if times else 0.0
        for s, times in strategy_data.items()
    }


def compute_average_oqs(results: list[dict]) -> dict[str, float]:
    """Strateji başına ortalama OQS hesaplar."""
    from collections import defaultdict
    strategy_data: dict[str, list] = defaultdict(list)
    for r in results:
        strategy_data[r["strategy"]].append(r.get("oqs", 0.0))
    return {
        s: sum(vals) / len(vals) if vals else 0.0
        for s, vals in strategy_data.items()
    }


# ─── Min-Max Normalizasyon ────────────────────────────────────────────────────
def minmax_normalize(values: dict[str, float], invert: bool = False) -> dict[str, float]:
    """
    Min-max normalizasyon uygular.
    invert=True: yüksek değer kötü ise (token, latency).
    """
    if not values:
        return {}
    vals = list(values.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 0.5 for k in values}  # Hepsi eşit
    normalized = {
        k: (v - min_v) / (max_v - min_v)
        for k, v in values.items()
    }
    if invert:
        return {k: 1 - v for k, v in normalized.items()}
    return normalized


# ─── CEI (Collaboration Efficiency Index) ─────────────────────────────────────
def compute_cei(
    results: list[dict],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    CEI = w1·TSR_norm + w2·OQS_norm − w3·TE_norm − w4·WCL_norm

    Args:
        results: Benchmark sonuç listesi.
        weights: {'w1': ..., 'w2': ..., 'w3': ..., 'w4': ...}
                 Varsayılan: w1=w2=w3=w4=0.25

    Returns:
        Strateji → CEI değeri sözlüğü.
    """
    if weights is None:
        weights = {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25}

    tsr = compute_tsr(results)
    oqs = compute_average_oqs(results)
    te = compute_token_expenditure(results)
    wcl = compute_wall_clock_latency(results)

    # Normalize
    tsr_norm = minmax_normalize(tsr, invert=False)
    oqs_norm = minmax_normalize(oqs, invert=False)
    te_norm = minmax_normalize(te, invert=True)   # Daha az token → daha iyi
    wcl_norm = minmax_normalize(wcl, invert=True)  # Daha az süre → daha iyi

    strategies = set(tsr.keys())
    cei = {}
    for s in strategies:
        ceil_val = (
            weights["w1"] * tsr_norm.get(s, 0)
            + weights["w2"] * oqs_norm.get(s, 0)
            + weights["w3"] * te_norm.get(s, 0)  # Pozitif çünkü invert edildi
            + weights["w4"] * wcl_norm.get(s, 0)
        )
        cei[s] = round(ceil_val, 4)

    return cei


def compute_all_cei_profiles(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    3 farklı ağırlık profili ile CEI hesaplar:
    - balanced: w1=w2=w3=w4=0.25
    - quality_focused: w1=0.4, w2=0.3, w3=0.15, w4=0.15
    - cost_focused: w1=0.2, w2=0.2, w3=0.3, w4=0.3
    """
    profiles = {
        "balanced": {"w1": 0.25, "w2": 0.25, "w3": 0.25, "w4": 0.25},
        "quality_focused": {"w1": 0.40, "w2": 0.30, "w3": 0.15, "w4": 0.15},
        "cost_focused": {"w1": 0.20, "w2": 0.20, "w3": 0.30, "w4": 0.30},
    }
    return {
        profile_name: compute_cei(results, weights)
        for profile_name, weights in profiles.items()
    }


# ─── Maliyet Hesabı ───────────────────────────────────────────────────────────
def compute_cost_per_success(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Strateji başına maliyet hesaplar.
    Cost = Total_tokens × Price_per_token
    Cost_per_success = Cost / Number_of_successes

    Varsayım: %30 input token, %70 output token (konuşma tabanlı sistem için)
    """
    from collections import defaultdict
    strategy_data: dict[str, dict] = defaultdict(lambda: {
        "total_tokens": 0, "successes": 0, "runs": 0
    })

    for r in results:
        s = r["strategy"]
        strategy_data[s]["total_tokens"] += r.get("total_tokens", 0)
        strategy_data[s]["successes"] += int(r.get("success", False))
        strategy_data[s]["runs"] += 1

    costs = {}
    for s, d in strategy_data.items():
        total_tokens = d["total_tokens"]
        input_tokens = int(total_tokens * 0.30)
        output_tokens = int(total_tokens * 0.70)

        total_cost_usd = (
            input_tokens * PRICE_PER_INPUT_TOKEN
            + output_tokens * PRICE_PER_OUTPUT_TOKEN
        )

        successes = d["successes"]
        cost_per_success = total_cost_usd / successes if successes > 0 else float("inf")

        costs[s] = {
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "successes": successes,
            "cost_per_success_usd": round(cost_per_success, 6),
        }

    return costs


# ─── Tam Değerlendirme Raporu ─────────────────────────────────────────────────
def full_evaluation_report(results: list[dict]) -> dict[str, Any]:
    """
    Tüm metrikleri hesaplayıp yapılandırılmış bir rapor döndürür.
    """
    tsr = compute_tsr(results)
    te = compute_token_expenditure(results)
    wcl = compute_wall_clock_latency(results)
    oqs = compute_average_oqs(results)
    cei_profiles = compute_all_cei_profiles(results)
    costs = compute_cost_per_success(results)

    report = {
        "tsr": tsr,
        "token_expenditure": te,
        "wall_clock_latency": wcl,
        "oqs": oqs,
        "cei": cei_profiles,
        "cost": costs,
    }
    return report


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    RESULTS_FILE = Path(__file__).parent / "results" / "benchmark_results.json"

    if not RESULTS_FILE.exists():
        print("⚠️  Benchmark sonuçları bulunamadı. Önce benchmark_runner.py'yi çalıştırın.")
        sys.exit(1)

    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    report = full_evaluation_report(results)

    print("=" * 60)
    print("BÖLÜM 4 — DEĞERLENDİRME RAPORU")
    print("=" * 60)

    print("\n📊 TSR (Task Success Rate):")
    for s, v in sorted(report["tsr"].items(), key=lambda x: -x[1]):
        print(f"  {s:<22}: {v:.1%}")

    print("\n📊 OQS (Ortalama Output Quality Score):")
    for s, v in sorted(report["oqs"].items(), key=lambda x: -x[1]):
        print(f"  {s:<22}: {v:.2f}/10")

    print("\n📊 CEI (Balanced):")
    for s, v in sorted(report["cei"]["balanced"].items(), key=lambda x: -x[1]):
        print(f"  {s:<22}: {v:.4f}")

    print("\n📊 Maliyet:")
    for s, v in report["cost"].items():
        print(f"  {s:<22}: ${v['total_cost_usd']:.6f} | CPS: ${v['cost_per_success_usd']:.6f}")

    # Raporu JSON'a kaydet
    REPORT_FILE = Path(__file__).parent / "results" / "evaluation_report.json"
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Rapor kaydedildi: {REPORT_FILE}")
