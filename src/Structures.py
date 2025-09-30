from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class Node:
    x: Optional[float] = None
    y: Optional[float] = None
    clusterId: Optional[int] = None

@dataclass
class Arc:
    src: int
    dest: int
    unit: float
    fixed: float
    capacity: float
    distance: Optional[float] = None
    time: Optional[int] = None

@dataclass
class Commodity:
    origin: int
    destination: int
    quantity: float
    availableTime: Optional[int] = None
    dueTime: Optional[int] = None

@dataclass
class ConfigParams:
    defaultInstancePath: str
    defaultNetworkPath: str
    folder: str
    networkNb: int
    instanceNb: int
    preProcessingSSNDP: bool
    doStatic: bool
    networkSeed: int
    demandSeed: int

@dataclass
class NetworkGeneratorParams:
    networkEmulationPath: str
    randomGeneration: bool
    capacity: float
    rectangleWidth: int
    rectangleHeight: int
    targetNodeNb: int
    targetArcNb: int
    targetDensity: int
    targetReciprocity: float
    decayRate: float
    hnRatio: float
    interArcsRatio: float
    connectedSpokesRatio: float
    ufCostRatio: float

    def __post_init__(self):
        if self.targetArcNb is None and self.targetDensity is None:
            raise TypeError("Value needed for at least one field: targetArcNb, targetDensity")
        
@dataclass
class InstanceGeneratorParams:
    # Static instance generation.
    doStatic: bool
    commodityNb: int
    quantityToCapacityMean: float
    quantityToCapacityDev: float
    # Timed instance generation.
    days: int
    hours: int
    flexibilityParameterMean: float
    flexibilityParameterDev: float
    criticalTime: int

class Network:
    def __init__(self, nodes=None, arcs=None):
        self.nodes: list[Node] = nodes if nodes is not None else []
        self.arcs: list[Arc] = arcs if arcs is not None else []

    @classmethod
    def from_file(cls, path: Path, isArcDistance: bool):
        nodes: list[Node] = []
        arcs: list[Arc] = []
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        if not lines[i].startswith("Nodes,"):
            raise ValueError("Expected 'Nodes,<count>' header at line 1")
        nodeNb = int(lines[i].split(",")[1])
        i += 1

        for _ in range(nodeNb):
            _, clusterId, x, y = lines[i].split(",")
            nodes.append(Node(float(x), float(y), clusterId))
            i += 1

        if not lines[i].startswith("Arcs,"):
            raise ValueError(f"Expected 'Arcs,<count>' header at line {i}")
        arcNb = int(lines[i].split(",")[1])
        i += 1

        for _ in range(arcNb):
            # id, src, dest, unit, fixed, capacity, distance
            _, src, dest, unit, fixed, capacity, joker = lines[i].split(",")
            time, distance = (None, joker) if isArcDistance else (joker, None)
            arcs.append(Arc(int(src), int(dest), float(unit), float(fixed), float(capacity), none_float_option(distance), none_float_option(time)))
            i += 1

        return cls(nodes, arcs)

class Instance:
    def __init__(self, nodes=None, arcs=None, commodities=None):
        self.nodes: list[Node] = nodes if nodes is not None else []
        self.arcs: list[Arc] = arcs if arcs is not None else []
        self.commodities: list[Commodity] = commodities if commodities is not None else []

    @classmethod
    def from_file(cls, path: Path):
        network = Network.from_file(path)
        nodeNb = len(network.nodes)
        arcNb = len(network.arcs)

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = nodeNb + arcNb + 1
        if not lines[i].startswith("Commodities,"):
            raise ValueError(f"Expected 'Commodities,<count>' header at line {i}")
        commodityNb = int(lines[i].split(",")[1])
        commodities: list[Commodity] = []
        for _ in range(commodityNb):
            # id, src, dest, quantity, available time, due time
            _, src, dest, quantity, availableTime, dueTime = lines[i].split(",")
            commodities.append(Commodity(int(src), int(dest), float(quantity), int(availableTime), int(dueTime)))
            i += 1

        return cls(network.nodes, network.arcs, commodities)

def none_float_option(x):
    return float(x) if x is not None else None