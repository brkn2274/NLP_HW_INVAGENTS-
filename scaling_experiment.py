"""
BONUS — Scaling Deneyi
Aynı stratejiyi 3, 5, 7 ajan ile çalıştırıp ajan sayısının etkisini analiz eder.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from bolum2_orkestrasyon.strategies import MajorityVotingStrategy, HierarchicalStrategy
from bolum4_degerlendirme.metrics import compute_oqs

RESULTS_DIR = ROOT / "bolum4_degerlendirme" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


SCALING_TASKS = [
    "Yapay zekanın etik riskleri nelerdir? Kapsamlı bir değerlendirme yapın.",
    "Küresel ısınmaya karşı alınabilecek 5 pratik önlem önerin ve her birini gerekçelendirin.",
    "Uzaktan çalışmanın verimliliğe etkisini birden fazla perspektiften tartışın.",
]


def run_scaling_experiment(strategy_cls, strategy_name: str, agent_counts: list[int]) -> list[dict]:
    """Belirli bir stratejiyi farklı ajan sayılarıyla çalıştırır."""
    all_results = []

    for n_agents in agent_counts:
        print(f"\n{'─'*50}")
        print(f"Strateji: {strategy_name} | Ajan Sayısı: {n_agents}")
        print("─" * 50)

        task_results = []
        for i, task in enumerate(SCALING_TASKS, 1):
            print(f"  Görev {i}: {task[:50]}...")
            start = time.time()

            # Strateji örneği oluştur
            if strategy_name == "majority_voting":
                strategy = strategy_cls(num_agents=n_agents)
            else:  # hierarchical
                strategy = strategy_cls(num_workers=n_agents)

            result = strategy.run(task)
            elapsed = time.time() - start

            oqs = compute_oqs(
                prediction=result.final_answer,
                task_prompt=task,
                tier=3,
            )

            record = {
                "strategy": strategy_name,
                "n_agents": n_agents,
                "task_index": i,
                "total_tokens": result.total_tokens,
                "elapsed_time_sec": round(elapsed, 4),
                "oqs": oqs,
            }
            task_results.append(record)
            print(f"    ✓ tokens={result.total_tokens} | OQS={oqs:.1f} | {elapsed:.2f}s")

        # Ortalama hesapla
        avg_tokens = sum(r["total_tokens"] for r in task_results) / len(task_results)
        avg_oqs = sum(r["oqs"] for r in task_results) / len(task_results)
        avg_time = sum(r["elapsed_time_sec"] for r in task_results) / len(task_results)

        summary = {
            "strategy": strategy_name,
            "n_agents": n_agents,
            "avg_tokens": round(avg_tokens, 1),
            "avg_oqs": round(avg_oqs, 2),
            "avg_time_sec": round(avg_time, 4),
            "task_details": task_results,
        }
        all_results.append(summary)

        print(f"  → Ortalama: OQS={avg_oqs:.2f} | Token={avg_tokens:.0f} | Süre={avg_time:.2f}s")

    return all_results


def main():
    print("=" * 60)
    print("BONUS — SCALING DENEYİ")
    print("Ajan sayısının etkisi: 3, 5, 7 ajan")
    print("=" * 60)

    agent_counts = [3, 5, 7]
    all_scaling_results = []

    # 1. Majority Voting scaling
    print("\n📊 Majority Voting Scaling:")
    mv_results = run_scaling_experiment(MajorityVotingStrategy, "majority_voting", agent_counts)
    all_scaling_results.extend(mv_results)

    # 2. Hierarchical scaling
    print("\n📊 Hierarchical Scaling:")
    hier_results = run_scaling_experiment(HierarchicalStrategy, "hierarchical", agent_counts)
    all_scaling_results.extend(hier_results)

    # ─── Özet Rapor ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCALING DENEYİ ÖZET")
    print("=" * 60)
    print(f"{'Strateji':<22} {'N_Ajan':>7} {'Avg Token':>10} {'Avg OQS':>9} {'Avg Süre':>10}")
    print("─" * 65)
    for r in all_scaling_results:
        print(f"{r['strategy']:<22} {r['n_agents']:>7} {r['avg_tokens']:>10.0f} "
              f"{r['avg_oqs']:>9.2f} {r['avg_time_sec']:>9.2f}s")

    # Analiz: diminishing returns tespiti
    print("\n📈 Ajan Sayısı Artışının Analizi:")
    for strat_name in ["majority_voting", "hierarchical"]:
        strat_results = [r for r in all_scaling_results if r["strategy"] == strat_name]
        if len(strat_results) >= 2:
            oqs_values = [r["avg_oqs"] for r in strat_results]
            token_values = [r["avg_tokens"] for r in strat_results]
            oqs_gain = oqs_values[-1] - oqs_values[0]
            token_cost = token_values[-1] - token_values[0]
            efficiency = oqs_gain / (token_cost + 1e-9) * 100

            print(f"\n  {strat_name}:")
            print(f"    OQS artışı (3→7 ajan): {oqs_gain:+.2f}")
            print(f"    Token artışı (3→7 ajan): {token_cost:+.0f}")
            print(f"    Verimlilik (OQS/token): {efficiency:.4f}")
            if oqs_gain < 0.5:
                print(f"    ⚠️  Azalan getiri (diminishing returns) gözlemlendi!")
            else:
                print(f"    ✓ Ajan sayısı artışı anlamlı kalite iyileştirmesi sağladı.")

    # ─── Kaydet ──────────────────────────────────────────────────────────────
    output_file = RESULTS_DIR / "scaling_experiment_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_scaling_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Scaling sonuçları kaydedildi: {output_file}")


if __name__ == "__main__":
    main()
