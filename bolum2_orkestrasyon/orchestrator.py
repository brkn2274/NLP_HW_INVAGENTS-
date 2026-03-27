"""
Bölüm 2 — Orkestratör
Strateji adı alır, uygun ajan konfigürasyonunu kurar ve çalıştırır.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .strategies import (
    BaseStrategy,
    OrchestratorResult,
    get_strategy,
    STRATEGY_REGISTRY,
)


LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Orchestrator:
    """
    Çoklu-ajan orkestrasyon motoru.
    Strateji adı alır, uygun konfigürasyonu kurar ve görevi çalıştırır.
    """

    def __init__(
        self,
        strategy_name: str,
        log_dir: Optional[Path] = None,
        strategy_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            strategy_name: Kullanılacak strateji adı (ör. "debate", "s4").
            log_dir: JSON loglarının kaydedileceği dizin.
            strategy_kwargs: Stratejiye iletilecek ek argümanlar.
        """
        self.strategy_name = strategy_name
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(exist_ok=True)
        self._strategy: BaseStrategy = get_strategy(
            strategy_name, **(strategy_kwargs or {})
        )

    def run(self, task: str, save_log: bool = True) -> OrchestratorResult:
        """
        Görevi seçili strateji ile çalıştırır.

        Args:
            task: Çözülecek görev metni.
            save_log: True ise sonucu JSON dosyasına kaydeder.

        Returns:
            OrchestratorResult nesnesi.
        """
        print(f"▶  Strateji: {self.strategy_name.upper()} | Görev: {task[:60]}...")
        start_wall = time.time()

        result = self._strategy.run(task)

        wall_time = time.time() - start_wall
        result.elapsed_time = wall_time

        print(f"   ✓ Tamamlandı — {wall_time:.2f}s | {result.total_tokens} token")

        if save_log:
            self._save_log(result)

        return result

    def _save_log(self, result: OrchestratorResult) -> Path:
        """Sonucu JSON formatında log dosyasına kaydeder."""
        timestamp = int(time.time())
        filename = f"{result.strategy}_{timestamp}.json"
        log_path = self.log_dir / filename

        log_data = result.to_dict()

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"   📄 Log kaydedildi: {log_path}")
        return log_path

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Kullanılabilir tüm strateji adlarını döndürür."""
        # Benzersiz sınıfları filtrele
        seen_classes = set()
        unique_names = []
        for name, cls_obj in STRATEGY_REGISTRY.items():
            if cls_obj not in seen_classes:
                seen_classes.add(cls_obj)
                unique_names.append(name)
        return unique_names

    def __repr__(self) -> str:
        return f"Orchestrator(strategy={self.strategy_name!r})"


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ORKESTRATÖR DEMO")
    print("=" * 60)

    task = "Yapay zeka sistemlerinde önyargı (bias) sorununu tartışın ve çözüm önerileri sunun."

    strategies = ["solo", "solo_refinement", "sequential_chain", "debate", "majority_voting"]

    for strat_name in strategies:
        print(f"\n{'─'*50}")
        orch = Orchestrator(strategy_name=strat_name)
        result = orch.run(task, save_log=False)
        print(f"   Nihai Yanıt (ilk 150 karakter): {result.final_answer[:150]}...")
        print(f"   Toplam Token: {result.total_tokens} | Süre: {result.elapsed_time:.3f}s")
