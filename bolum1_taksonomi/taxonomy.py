"""
Bölüm 1 — Taksonomi Modelleme
4-boyutlu LLM Çoklu-Ajan Sistem Taksonomisi
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional


# ─── 1. Boyut: Topology ────────────────────────────────────────────────────────
class Topology(Enum):
    """Çoklu-ajan sisteminin topolojik yapısı."""
    CENTRALIZED = "centralized"
    DECENTRALIZED_FLAT = "decentralized_flat"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"


# ─── 2. Boyut: Communication Protocol ─────────────────────────────────────────
class CommunicationProtocol(Enum):
    """Ajanlar arası iletişim protokolü."""
    NATURAL_LANGUAGE = "natural_language"
    STRUCTURED_ARTIFACTS = "structured_artifacts"
    SHARED_MEMORY = "shared_memory"
    EVENT_BUS = "event_bus"


# ─── 3. Boyut: Conflict Resolution ────────────────────────────────────────────
class ConflictResolution(Enum):
    """Ajanlar arasındaki çelişkileri çözme mekanizması."""
    ARBITRATOR = "arbitrator"
    VOTING = "voting"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    METACOGNITIVE = "metacognitive"


# ─── 4. Boyut: Task Decomposition ─────────────────────────────────────────────
class TaskDecomposition(Enum):
    """Görev ayrıştırma stratejisi."""
    NONE = "none"
    PREDEFINED = "predefined"
    LLM_PLANNED = "llm_planned"
    ADAPTIVE = "adaptive"


# ─── Strategy Dataclass ────────────────────────────────────────────────────────
@dataclass
class Strategy:
    """Dört boyutun kombinasyonunu temsil eden çoklu-ajan stratejisi."""
    name: str
    topology: Topology
    communication: CommunicationProtocol
    resolution: ConflictResolution
    decomposition: TaskDecomposition
    description: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"Strategy(name={self.name!r}, "
            f"topology={self.topology}, "
            f"communication={self.communication}, "
            f"resolution={self.resolution}, "
            f"decomposition={self.decomposition})"
        )


# ─── S1–S9 Stratejileri ────────────────────────────────────────────────────────
STRATEGIES: dict[str, Strategy] = {
    "S1": Strategy(
        name="S1",
        topology=Topology.CENTRALIZED,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.NONE,
        description="Solo Agent — Tek ajan, tek seferde yanıt üretir.",
    ),
    "S1+": Strategy(
        name="S1+",
        topology=Topology.CENTRALIZED,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.METACOGNITIVE,
        decomposition=TaskDecomposition.NONE,
        description="Solo + Self-Refinement — Tek ajan çıktısını iteratif olarak iyileştirir.",
    ),
    "S2": Strategy(
        name="S2",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.SHARED_MEMORY,
        resolution=ConflictResolution.CONSENSUS,
        decomposition=TaskDecomposition.NONE,
        description="Parallel Independent — Ajanlar paylaşımlı bellekle bağımsız çalışır.",
    ),
    "S3": Strategy(
        name="S3",
        topology=Topology.CENTRALIZED,
        communication=CommunicationProtocol.STRUCTURED_ARTIFACTS,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.PREDEFINED,
        description="Sequential Chain — Ajanlar sırayla çalışır, önceki çıktı girdiye dönüşür.",
    ),
    "S4": Strategy(
        name="S4",
        topology=Topology.HIERARCHICAL,
        communication=CommunicationProtocol.STRUCTURED_ARTIFACTS,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.LLM_PLANNED,
        description="Hierarchical — Lider ajan alt-ajanlara görev dağıtır ve sonuçları birleştirir.",
    ),
    "S5": Strategy(
        name="S5",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.DEBATE,
        decomposition=TaskDecomposition.NONE,
        description="Debate — İki ajan karşılıklı argüman üretir, hakem karar verir.",
    ),
    "S6": Strategy(
        name="S6",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.VOTING,
        decomposition=TaskDecomposition.NONE,
        description="Majority Voting — N ajan bağımsız yanıt verir, çoğunluk oyuyla seçilir.",
    ),
    "S7": Strategy(
        name="S7",
        topology=Topology.HIERARCHICAL,
        communication=CommunicationProtocol.EVENT_BUS,
        resolution=ConflictResolution.CONSENSUS,
        decomposition=TaskDecomposition.ADAPTIVE,
        description="Adaptive Hierarchical — Dinamik görev dağıtımı ve olay tabanlı iletişim.",
    ),
    "S8": Strategy(
        name="S8",
        topology=Topology.DYNAMIC,
        communication=CommunicationProtocol.EVENT_BUS,
        resolution=ConflictResolution.METACOGNITIVE,
        decomposition=TaskDecomposition.ADAPTIVE,
        description="Dynamic Metacognitive — Sistem kendi performansını değerlendirip strateji değiştirir.",
    ),
    "S9": Strategy(
        name="S9",
        topology=Topology.DYNAMIC,
        communication=CommunicationProtocol.SHARED_MEMORY,
        resolution=ConflictResolution.CONSENSUS,
        decomposition=TaskDecomposition.LLM_PLANNED,
        description="Swarm Consensus — Çok sayıda hafif ajan paylaşımlı bellek üzerinde uzlaşı sağlar.",
    ),
}


# ─── Framework → Strategy Eşleştirme ──────────────────────────────────────────
FRAMEWORK_MAP: dict[str, Strategy] = {
    "MetaGPT": Strategy(
        name="MetaGPT",
        topology=Topology.HIERARCHICAL,
        communication=CommunicationProtocol.STRUCTURED_ARTIFACTS,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.PREDEFINED,
        description="MetaGPT: Rol tabanlı hiyerarşik ajan sistemi, SOP (Standard Operating Procedure) kullanır.",
    ),
    "AutoGen": Strategy(
        name="AutoGen",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.CONSENSUS,
        decomposition=TaskDecomposition.LLM_PLANNED,
        description="AutoGen: Çok-ajanli konuşma çerçevesi, esnek ajan etkileşimleri.",
    ),
    "CrewAI": Strategy(
        name="CrewAI",
        topology=Topology.HIERARCHICAL,
        communication=CommunicationProtocol.STRUCTURED_ARTIFACTS,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.PREDEFINED,
        description="CrewAI: Rol ve görev tabanlı ajan ekipleri, sıralı ya da hiyerarşik işleyiş.",
    ),
    "LangGraph": Strategy(
        name="LangGraph",
        topology=Topology.DYNAMIC,
        communication=CommunicationProtocol.SHARED_MEMORY,
        resolution=ConflictResolution.METACOGNITIVE,
        decomposition=TaskDecomposition.ADAPTIVE,
        description="LangGraph: Durum makinesi tabanlı ajan orkestrasyonu, döngüsel graf yapısı.",
    ),
    "AutoGPT": Strategy(
        name="AutoGPT",
        topology=Topology.CENTRALIZED,
        communication=CommunicationProtocol.STRUCTURED_ARTIFACTS,
        resolution=ConflictResolution.METACOGNITIVE,
        decomposition=TaskDecomposition.LLM_PLANNED,
        description="AutoGPT: Tek ajan özyinelemeli planlama ve araç kullanımı.",
    ),
    "BabyAGI": Strategy(
        name="BabyAGI",
        topology=Topology.CENTRALIZED,
        communication=CommunicationProtocol.SHARED_MEMORY,
        resolution=ConflictResolution.ARBITRATOR,
        decomposition=TaskDecomposition.LLM_PLANNED,
        description="BabyAGI: Görev listesi tabanlı otonom ajan, öncelik kuyruklu planlama.",
    ),
    "OpenAgents": Strategy(
        name="OpenAgents",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.EVENT_BUS,
        resolution=ConflictResolution.VOTING,
        decomposition=TaskDecomposition.ADAPTIVE,
        description="OpenAgents: Uzman ajan havuzu, kullanıcı ihtiyacına göre dinamik ajan seçimi.",
    ),
    "Camel": Strategy(
        name="Camel",
        topology=Topology.DECENTRALIZED_FLAT,
        communication=CommunicationProtocol.NATURAL_LANGUAGE,
        resolution=ConflictResolution.DEBATE,
        decomposition=TaskDecomposition.PREDEFINED,
        description="Camel: Rol yapma tabanlı iletişimli ajan etkileşimi, çift-ajan tartışması.",
    ),
}


def classify(framework_name: str) -> Strategy:
    """
    Verilen framework adı için taksonomi eşleştirmesini döndürür.

    Args:
        framework_name: Framework adı (büyük/küçük harf duyarsız).

    Returns:
        Strategy dataclass örneği.

    Raises:
        ValueError: Bilinmeyen framework adı.
    """
    # Büyük/küçük harf normalizasyonu
    normalized = framework_name.strip()
    for key, strategy in FRAMEWORK_MAP.items():
        if key.lower() == normalized.lower():
            return strategy
    supported = ", ".join(FRAMEWORK_MAP.keys())
    raise ValueError(
        f"Bilinmeyen framework: {framework_name!r}. "
        f"Desteklenen frameworkler: {supported}"
    )


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("BÖLÜM 1 — TAKSONOMİ MODELLEMESİ")
    print("=" * 60)

    print("\n📌 Enum Değerleri:")
    for enum_cls in [Topology, CommunicationProtocol, ConflictResolution, TaskDecomposition]:
        print(f"  {enum_cls.__name__}: {[e.value for e in enum_cls]}")

    print("\n📌 S1–S9 Stratejileri:")
    for key, s in STRATEGIES.items():
        print(f"  [{key}] {s.description}")

    print("\n📌 Framework Sınıflandırmaları:")
    for fw in ["MetaGPT", "AutoGen", "CrewAI", "LangGraph", "AutoGPT", "Camel"]:
        print(f"\n  >>> classify({fw!r})")
        print(f"  {classify(fw)}")
