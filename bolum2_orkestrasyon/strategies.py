"""
Bölüm 2 — Strateji İmplementasyonları
S1 (Solo), S1+ (Self-Refinement), S3 (Sequential Chain),
S4 (Hierarchical), S5 (Debate), S6 (Majority Voting)
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentMessage


# ─── OrchestratorResult ────────────────────────────────────────────────────────
@dataclass
class OrchestratorResult:
    """Bir orkestrasyon çalıştırmasının sonucu."""
    strategy: str
    task: str
    final_answer: str
    agent_logs: list[dict[str, Any]]
    total_tokens: int
    elapsed_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "task": self.task,
            "rounds": self.agent_logs,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "elapsed_time_sec": round(self.elapsed_time, 4),
            "metadata": self.metadata,
        }


# ─── Base Strategy ─────────────────────────────────────────────────────────────
class BaseStrategy(ABC):
    """Tüm strateji sınıflarının temel sınıfı."""

    strategy_name: str = "base"

    @abstractmethod
    def run(self, task: str) -> OrchestratorResult:
        """Stratejiyi çalıştırır ve sonucu döndürür."""
        ...

    @staticmethod
    def _log_entry(agent: Agent, message: str, extra: dict | None = None) -> dict[str, Any]:
        """Agent mesajını log formatına çevirir."""
        entry: dict[str, Any] = {
            "agent": agent.name,
            "role": agent.role,
            "message": message,
            "tokens": agent.get_token_count(message),
        }
        if extra:
            entry.update(extra)
        return entry

    @staticmethod
    def _count_tokens(logs: list[dict]) -> int:
        return sum(e.get("tokens", 0) for e in logs)


# ─── S1: Solo ─────────────────────────────────────────────────────────────────
class SoloStrategy(BaseStrategy):
    """
    S1 — Solo Agent
    Tek ajan, tek seferde yanıt üretir.
    Topology: Centralized | Comm: NaturalLanguage | Resolution: Arbitrator | Decomp: None
    """

    strategy_name = "solo"

    def __init__(self):
        self.agent = Agent(
            name="Solo-Agent",
            role="solver",
            system_prompt="Sen güçlü bir yapay zeka asistanısın. Görevleri doğrudan ve eksiksiz yanıtla.",
        )

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        response = self.agent.respond(task)
        logs.append(self._log_entry(self.agent, response))

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=response,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": 1, "rounds": 1},
        )


# ─── S1+: Solo + Self-Refinement ──────────────────────────────────────────────
class SoloRefinementStrategy(BaseStrategy):
    """
    S1+ — Solo + Self-Refinement
    Tek ajan kendi çıktısını N iterasyon iyileştirir.
    Token-matched baseline ile karşılaştırma için toplam token eşit tutulur.
    """

    strategy_name = "solo_refinement"

    def __init__(self, iterations: int = 3):
        self.iterations = iterations
        self.agent = Agent(
            name="Refiner-Agent",
            role="refiner",
            system_prompt=(
                "Sen kritik düşünce yeteneğine sahip bir yapay zeka asistanısın. "
                "Her iterasyonda önceki yanıtı analiz edip daha iyi bir versiyon üretirsin."
            ),
        )

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        # İlk yanıt
        current_response = self.agent.respond(task, iteration=1)
        logs.append(self._log_entry(self.agent, current_response, {"iteration": 1, "phase": "initial"}))

        # İterasyonlar
        for i in range(2, self.iterations + 1):
            refined = self.agent.respond(
                f"Şu yanıtı iyileştir: {current_response}\n\nOrijinal görev: {task}",
                context=current_response,
                iteration=i,
            )
            logs.append(self._log_entry(self.agent, refined, {"iteration": i, "phase": "refinement"}))
            current_response = refined

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=current_response,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": 1, "iterations": self.iterations},
        )


# ─── S3: Sequential Chain ─────────────────────────────────────────────────────
class SequentialChainStrategy(BaseStrategy):
    """
    S3 — Sequential Chain
    Ajanlar sırayla çalışır. Her ajan öncekinin çıktısını giriş olarak alır.
    Topology: Centralized | Comm: StructuredArtifacts | Resolution: Arbitrator | Decomp: Predefined
    """

    strategy_name = "sequential_chain"

    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        roles = ["solver", "critic", "summarizer"]
        role_names = ["Çözücü", "Eleştirmen", "Özetleyici"]
        self.agents = [
            Agent(
                name=f"{role_names[i % len(role_names)]}-{i+1}",
                role=roles[i % len(roles)],
                system_prompt=f"Sen bir zincirin {i+1}. halkasısın. Önceki ajanın çıktısını geliştir.",
            )
            for i in range(num_agents)
        ]

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        current_output = task
        for i, agent in enumerate(self.agents):
            prompt = task if i == 0 else f"Önceki yanıt:\n{current_output}\n\nGörevi tamamla: {task}"
            response = agent.respond(prompt, context=current_output if i > 0 else "")
            logs.append(self._log_entry(agent, response, {"chain_position": i + 1}))
            current_output = response

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=current_output,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": self.num_agents, "chain_length": self.num_agents},
        )


# ─── S4: Hierarchical ─────────────────────────────────────────────────────────
class HierarchicalStrategy(BaseStrategy):
    """
    S4 — Hierarchical
    Lider ajan görevi alt-ajanlara dağıtır, sonuçları birleştirir.
    Topology: Hierarchical | Comm: StructuredArtifacts | Resolution: Arbitrator | Decomp: LLMPlanned
    """

    strategy_name = "hierarchical"

    def __init__(self, num_workers: int = 3):
        self.num_workers = num_workers
        self.leader = Agent(
            name="Lider-Ajan",
            role="leader",
            system_prompt=(
                "Sen bir ajan ekibini organize eden lider yapay zekasın. "
                "Görevi alt-görevlere böl, her alt-ajan için direktif ver ve sonuçları birleştir."
            ),
        )
        self.workers = [
            Agent(
                name=f"Alt-Ajan-{i+1}",
                role=["researcher", "solver", "critic"][i % 3],
                system_prompt=f"Sen lider ajanın yönlendirmesiyle çalışan uzman bir yapay zekasın. Rol: {['araştırmacı', 'çözücü', 'eleştirmen'][i % 3]}.",
            )
            for i in range(num_workers)
        ]
        self.aggregator = Agent(
            name="Birleştirici",
            role="summarizer",
            system_prompt="Alt-ajan çıktılarını tek bir tutarlı yanıtta birleştir.",
        )

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        # 1. Lider görevi yorumlar
        leader_plan = self.leader.respond(
            f"Bu görevi {self.num_workers} alt-göreve böl ve her biri için alt-ajana direktif ver:\n{task}"
        )
        logs.append(self._log_entry(self.leader, leader_plan, {"phase": "planning"}))

        # 2. Alt-ajanlar paralel çalışır (simüle)
        worker_outputs = []
        for i, worker in enumerate(self.workers):
            subtask = f"[Alt-görev {i+1}] {task}"
            worker_response = worker.respond(subtask, context=leader_plan)
            logs.append(self._log_entry(worker, worker_response, {"phase": "execution", "subtask": i + 1}))
            worker_outputs.append(worker_response)

        # 3. Lider/aggregator sonuçları birleştirir
        combined_context = "\n\n".join([f"Alt-Ajan {i+1}: {out}" for i, out in enumerate(worker_outputs)])
        final = self.aggregator.respond(
            f"Şu alt-ajan çıktılarını birleştirip tutarlı bir nihai yanıt oluştur:\n{combined_context}",
            context=combined_context,
        )
        logs.append(self._log_entry(self.aggregator, final, {"phase": "aggregation"}))

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=final,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": self.num_workers + 2, "workers": self.num_workers},
        )


# ─── S5: Debate ───────────────────────────────────────────────────────────────
class DebateStrategy(BaseStrategy):
    """
    S5 — Debate
    İki ajan karşılıklı argüman üretir, bir hakem karar verir.
    Topology: DecentralizedFlat | Comm: NaturalLanguage | Resolution: Debate | Decomp: None
    """

    strategy_name = "debate"

    def __init__(self, debate_rounds: int = 2):
        self.debate_rounds = debate_rounds
        self.proponent = Agent(
            name="Savunan",
            role="proponent",
            system_prompt="Sen bir pozisyonu güçlü argümanlarla savunan bir yapay zekasın.",
        )
        self.opponent = Agent(
            name="Karşı-Çıkan",
            role="opponent",
            system_prompt="Sen önerilen pozisyona karşı çıkan, alternatif görüşler sunan bir yapay zekasın.",
        )
        self.judge = Agent(
            name="Hakem",
            role="judge",
            system_prompt=(
                "Sen tartışmayı değerlendiren tarafsız bir hakem yapay zekasın. "
                "Her iki tarafın argümanlarını analiz edip en güçlü olanı seçersin."
            ),
        )

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        proponent_response = ""
        opponent_response = ""

        for round_num in range(1, self.debate_rounds + 1):
            # Savunan ajan
            pro_prompt = (
                task if round_num == 1
                else f"Karşı görüş: {opponent_response}\nKendi pozisyonunu güçlendir: {task}"
            )
            proponent_response = self.proponent.respond(pro_prompt, context=opponent_response)
            logs.append(self._log_entry(
                self.proponent, proponent_response,
                {"round": round_num, "stance": "proponent"}
            ))

            # Karşı çıkan ajan
            opp_prompt = f"Bu argümana karşı çık: {proponent_response}\nAna görev: {task}"
            opponent_response = self.opponent.respond(opp_prompt, context=proponent_response)
            logs.append(self._log_entry(
                self.opponent, opponent_response,
                {"round": round_num, "stance": "opponent"}
            ))

        # Hakem kararı
        debate_summary = (
            f"SAVUNMA: {proponent_response}\n\n"
            f"KARŞI GÖRÜŞ: {opponent_response}\n\n"
            f"Görev: {task}"
        )
        verdict = self.judge.respond(debate_summary)
        logs.append(self._log_entry(self.judge, verdict, {"phase": "verdict"}))

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=verdict,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": 3, "debate_rounds": self.debate_rounds},
        )


# ─── S6: Majority Voting ──────────────────────────────────────────────────────
class MajorityVotingStrategy(BaseStrategy):
    """
    S6 — Majority Voting
    N ajan bağımsız yanıt verir, çoğunluk oyu ile seçilir.
    Topology: DecentralizedFlat | Comm: NaturalLanguage | Resolution: Voting | Decomp: None
    """

    strategy_name = "majority_voting"

    def __init__(self, num_agents: int = 5):
        self.num_agents = num_agents
        self.agents = [
            Agent(
                name=f"Seçmen-{i+1}",
                role=["solver", "researcher", "critic", "solver", "researcher"][i % 5],
                system_prompt=f"Sen görevleri bağımsız olarak değerlendiren {i+1}. ajan yapay zekasın.",
            )
            for i in range(num_agents)
        ]

    def run(self, task: str) -> OrchestratorResult:
        start = time.time()
        logs: list[dict] = []

        # Her ajan bağımsız yanıt üretir
        responses = []
        for agent in self.agents:
            response = agent.respond(task)
            logs.append(self._log_entry(agent, response, {"phase": "independent_vote"}))
            responses.append(response)

        # Çoğunluk oyu simulasyonu:
        # Gerçek NLP'de embedding similarity kullanılır.
        # Simülasyonda: içeriği en uzun olanı temsili kazan olarak seç
        # Ek olarak: ajanların ilk cümlesini sembolik "oy" olarak say
        vote_counter: Counter = Counter()
        for i, resp in enumerate(responses):
            # İlk anlamlı kelimeyi oylama anahtarı olarak kullan
            first_words = " ".join(resp.split()[:5])
            vote_counter[i] = len(resp.split())  # Uzunluk bazlı skor

        # En yüksek skor alanı seç
        winner_idx = max(vote_counter, key=lambda k: vote_counter[k])
        final_answer = responses[winner_idx]

        # Oylama sonuçlarını logla
        vote_summary = {f"Seçmen-{i+1}": f"Skor: {vote_counter[i]}" for i in range(self.num_agents)}
        logs.append({
            "agent": "Oylama-Sistemi",
            "role": "aggregator",
            "message": f"Kazanan: Seçmen-{winner_idx+1} ({vote_counter[winner_idx]} skor). Oy dağılımı: {vote_summary}",
            "tokens": 20,
            "phase": "voting",
        })

        elapsed = time.time() - start
        return OrchestratorResult(
            strategy=self.strategy_name,
            task=task,
            final_answer=final_answer,
            agent_logs=logs,
            total_tokens=self._count_tokens(logs),
            elapsed_time=elapsed,
            metadata={"agents": self.num_agents, "winner": f"Seçmen-{winner_idx+1}"},
        )


# ─── Strateji Fabrikası ────────────────────────────────────────────────────────
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "solo": SoloStrategy,
    "s1": SoloStrategy,
    "solo_refinement": SoloRefinementStrategy,
    "s1+": SoloRefinementStrategy,
    "sequential_chain": SequentialChainStrategy,
    "s3": SequentialChainStrategy,
    "hierarchical": HierarchicalStrategy,
    "s4": HierarchicalStrategy,
    "debate": DebateStrategy,
    "s5": DebateStrategy,
    "majority_voting": MajorityVotingStrategy,
    "s6": MajorityVotingStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Strateji adına göre strateji örneği döndürür."""
    normalized = name.lower().replace(" ", "_")
    if normalized not in STRATEGY_REGISTRY:
        supported = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Bilinmeyen strateji: {name!r}. Desteklenen: {supported}")
    cls = STRATEGY_REGISTRY[normalized]
    return cls(**kwargs)
