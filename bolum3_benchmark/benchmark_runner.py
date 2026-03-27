"""
Bölüm 3 — Benchmark Runner
Tüm görevleri tüm stratejiler üzerinde çalıştırır, sonuçları CSV/JSON olarak kaydeder.
"""

from __future__ import annotations
import json
import csv
import sys
import time
import os
from pathlib import Path
from typing import Any

# Path düzeltmesi
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bolum2_orkestrasyon.strategies import (
    SoloStrategy,
    SoloRefinementStrategy,
    SequentialChainStrategy,
    HierarchicalStrategy,
    DebateStrategy,
    MajorityVotingStrategy,
)
from bolum4_degerlendirme.metrics import evaluate_result

TASKS_FILE = Path(__file__).parent / "tasks.json"
RESULTS_DIR = ROOT / "bolum4_degerlendirme" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_tasks() -> list[dict[str, Any]]:
    """tasks.json dosyasından görevleri yükler."""
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_strategies() -> dict[str, Any]:
    """Çalıştırılacak strateji sözlüğünü oluşturur."""
    return {
        "solo": SoloStrategy(),
        "solo_refinement": SoloRefinementStrategy(iterations=3),
        "sequential_chain": SequentialChainStrategy(num_agents=3),
        "hierarchical": HierarchicalStrategy(num_workers=3),
        "debate": DebateStrategy(debate_rounds=2),
        "majority_voting": MajorityVotingStrategy(num_agents=5),
    }


def run_benchmark(
    tasks: list[dict] | None = None,
    strategies: dict | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Tüm görevleri tüm stratejiler üzerinde çalıştırır.

    Args:
        tasks: Görev listesi (None ise tasks.json'dan yükler).
        strategies: Strateji sözlüğü (None ise varsayılan set kullanılır).
        verbose: True ise ilerleme yazdırır.

    Returns:
        Her çalıştırma için sonuç kayıtlarının listesi.
    """
    if tasks is None:
        tasks = load_tasks()
    if strategies is None:
        strategies = build_strategies()

    results: list[dict[str, Any]] = []
    total_runs = len(tasks) * len(strategies)
    run_num = 0

    if verbose:
        print("=" * 70)
        print(f"BENCHMARK BAŞLADI — {len(tasks)} görev × {len(strategies)} strateji = {total_runs} çalıştırma")
        print("=" * 70)

    for task in tasks:
        task_id = task["task_id"]
        tier = task["tier"]
        prompt = task["prompt"]
        expected = task.get("expected_answer")
        rubric = task.get("rubric")
        domain = task.get("domain", "general")

        for strat_name, strategy in strategies.items():
            run_num += 1
            if verbose:
                print(f"\n[{run_num}/{total_runs}] {task_id} × {strat_name}")

            try:
                result = strategy.run(prompt)

                # Metrik hesapla
                metrics = evaluate_result(
                    result=result,
                    expected_answer=expected,
                    rubric=rubric,
                    tier=tier,
                )

                record = {
                    "task_id": task_id,
                    "tier": tier,
                    "domain": domain,
                    "strategy": strat_name,
                    "success": metrics["success"],
                    "tsr": metrics["tsr"],
                    "oqs": metrics["oqs"],
                    "total_tokens": result.total_tokens,
                    "elapsed_time_sec": round(result.elapsed_time, 4),
                    "final_answer_preview": result.final_answer[:120].replace("\n", " "),
                    "agent_count": result.metadata.get("agents", 1),
                }
                results.append(record)

                if verbose:
                    status = "✓" if metrics["success"] else "✗"
                    print(f"   {status} success={metrics['success']} | OQS={metrics['oqs']:.1f} | "
                          f"tokens={result.total_tokens} | {result.elapsed_time:.2f}s")

            except Exception as e:
                if verbose:
                    print(f"   ❌ HATA: {e}")
                results.append({
                    "task_id": task_id,
                    "tier": tier,
                    "domain": domain,
                    "strategy": strat_name,
                    "success": False,
                    "tsr": 0.0,
                    "oqs": 0.0,
                    "total_tokens": 0,
                    "elapsed_time_sec": 0.0,
                    "final_answer_preview": f"ERROR: {str(e)}",
                    "agent_count": 0,
                    "error": str(e),
                })

    # ─── Kaydet ───────────────────────────────────────────────────────────────
    _save_csv(results)
    _save_json(results)

    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARK TAMAMLANDI — {len(results)} sonuç kaydedildi.")
        print(f"Sonuçlar: {RESULTS_DIR}")

    return results


def _save_csv(results: list[dict]) -> Path:
    """Sonuçları CSV olarak kaydeder."""
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not results:
        return csv_path
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"   💾 CSV: {csv_path}")
    return csv_path


def _save_json(results: list[dict]) -> Path:
    """Sonuçları JSON olarak kaydeder."""
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"   💾 JSON: {json_path}")
    return json_path


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_benchmark(verbose=True)

    # Özet istatistik
    from collections import defaultdict
    strat_summary: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0, "tokens": 0, "time": 0.0})
    for r in results:
        s = r["strategy"]
        strat_summary[s]["total"] += 1
        strat_summary[s]["success"] += int(r["success"])
        strat_summary[s]["tokens"] += r["total_tokens"]
        strat_summary[s]["time"] += r["elapsed_time_sec"]

    print("\n📊 STRATEJI ÖZETİ:")
    print(f"{'Strateji':<22} {'TSR':>6} {'Ort.Token':>10} {'Ort.Süre':>10}")
    print("─" * 55)
    for strat, data in strat_summary.items():
        tsr = data["success"] / data["total"] if data["total"] > 0 else 0
        avg_tok = data["tokens"] / data["total"] if data["total"] > 0 else 0
        avg_time = data["time"] / data["total"] if data["total"] > 0 else 0
        print(f"{strat:<22} {tsr:>6.1%} {avg_tok:>10.0f} {avg_time:>9.2f}s")
