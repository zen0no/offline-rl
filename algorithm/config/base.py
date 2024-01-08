from dataclasses import dataclass, field



@dataclass
class WandbConfig:
    project: str = 'Bellman-Wasserstein-distance(ICLR)'
