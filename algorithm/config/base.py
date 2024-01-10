from dataclasses import dataclass, field



@dataclass
class WandbConfig:
    project: str = 'Bellman-Wasserstein-distance(ICLR)'
    name: str = 'project_name'
    group: str = 'group_name'