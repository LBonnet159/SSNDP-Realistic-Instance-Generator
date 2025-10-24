from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx

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
    networkSeed: int
    demandSeed: int

    def __str__(self):
        return " | ".join(f"{k}={v}" for k, v in asdict(self).items())

@dataclass
class NetworkGeneratorParams:
    networkEmulationPath: str
    networkEmulationTimeLimit: float
    randomGeneration: bool
    capacity: float
    bboxWidth: int
    bboxHeight: int
    targetNodeNb: int
    targetArcNb: int
    targetDensity: int
    targetReciprocity: float
    decayRate: float
    hnRatio: float
    ufCostRatio: float

    def __post_init__(self):
        # networkEmulationPath + networkEmulationTimeLimit
        if self.networkEmulationPath is not None and self.networkEmulationTimeLimit is None:
            raise ValueError("Parameter networkEmulationTimeLimit must be specified if a network is being emulated.")

        # randomGeneration
        if self.randomGeneration is None:
            raise ValueError("Parameter randomGeneration must be either True or False.")

        # capacity
        if self.capacity is None:
            raise ValueError("Parameter capacity must be specified.")
        validate_value(self.capacity, lambda v: v > 0, "Parameter capacity value must be >0.")

        # bboxWidth & bboxHeight
        if not self.randomGeneration:
            if self.bboxWidth is None:
                raise ValueError("Parameter bboxWidth must be specified for non-random networks.")
            if self.bboxHeight is None:
                raise ValueError("Parameter bboxHeight must be specified for non-random networks.")
        validate_value(self.bboxWidth, lambda v: v > 0, "Parameter bboxWidth value must be >0.")
        validate_value(self.bboxHeight, lambda v: v > 0, "Parameter bboxHeight value must be >0.")

        # targetNodeNb
        if self.targetNodeNb is None:
            raise ValueError("Parameter targetNodeNb must be specified.")
        validate_value(self.targetNodeNb, lambda v: v > 0, "Parameter targetNodeNb value must be >0.")

        # targetArcNb + targetDensity
        if self.targetArcNb is None and self.targetDensity is None:
            raise ValueError("Value needed for at least one parameter: targetArcNb, targetDensity")

        # targetArcNb
        if self.targetArcNb is not None:
            validate_value(self.targetArcNb, lambda v: v > 0, "Parameter targetArcNb value must be >0.")
            maxArcNb = self.targetNodeNb * (self.targetNodeNb - 1)
            validate_value(self.targetArcNb, lambda v: v <= maxArcNb, 
                           f"Parameter targetArcNb must be ≤ maximum possible directed arcs ({maxArcNb}).")

        # targetDensity
        validate_value(self.targetDensity, lambda v: 0.0 <= v <= 1.0, "Parameter targetDensity value must be in the range [0,1].")

        # targetReciprocity
        if not self.randomGeneration and self.targetReciprocity is None:
            raise ValueError("Parameter targetReciprocity must be specified for non-random networks.")
        validate_value(self.targetReciprocity, lambda v: 0.0 <= v <= 1.0, "Parameter targetReciprocity value must be in the range [0,1].")

        # hnRatio
        if not self.randomGeneration and self.hnRatio is None:
            raise ValueError("Parameter hnRatio must be specified for non-random networks.")
        validate_value(self.hnRatio, lambda v: 0.0 < v <= 1.0, "Parameter hnRatio value must be in the range ]0,1].")

        # ufCostRatio
        if not self.randomGeneration and self.ufCostRatio is None:
            raise ValueError("Parameter ufCostRatio must be specified for non-random networks.")
        validate_value(self.ufCostRatio, lambda v: v > 0, "Parameter ufCostRatio value must be >0.")
        
    def __str__(self):
        return " | ".join(f"{k}={v}" for k, v in asdict(self).items())
        
@dataclass
class InstanceGeneratorParams:
    # Static instance generation.
    doStatic: bool
    commodityNb: int
    quantityToCapaMean: float
    quantityToCapaDev: float
    sameRegionRatio: float
    disparityRatio: float
    # Timed instance generation.
    horizon: int
    flexibilityMean: float
    flexibilityDev: float
    criticalTime: int
    distributionPattern: list[float]
    preProcessingSSNDP: bool

    def __post_init__(self):
        # doStatic
        if self.doStatic is None:
            raise ValueError("Parameter doStatic must be either True or False.")
        
        # commodityNb
        if self.commodityNb is None:
            raise ValueError("Parameter commodityNb must be specified.")
        validate_value(self.commodityNb, lambda v: v > 0, "Parameter commodityNb must be >0.")
        
        # quantityToCapaMean
        if self.quantityToCapaMean is None:
            raise ValueError("Parameter quantityToCapaMean must be specified.")
        validate_value(self.quantityToCapaMean, lambda v: v > 0, "Parameter quantityToCapaMean value must be >0.")

        # quantityToCapaDev
        if self.quantityToCapaDev is None:
            raise ValueError("Parameter quantityToCapaDev must be specified.")
        validate_value(self.quantityToCapaDev, lambda v: v >= 0, "Parameter quantityToCapaDev value must be >=0.")

        # sameRegionRatio
        if self.sameRegionRatio is not None:
            validate_value(self.sameRegionRatio, lambda v: 0 <= v <= 1, "Parameter sameRegionRatio must be in [0,1].")

        # disparityRatio
        if self.disparityRatio is not None:
            validate_value(self.disparityRatio, lambda v: 0 <= v <= 1, "Parameter disparityRatio must be in [0,1].")

        # SSNDP
        if not self.doStatic:
            # horizon
            if self.horizon is None:
                raise ValueError("Parameter horizon must be specified when generating a SSNDP instance.")
            validate_value(self.horizon, lambda v: v > 0, "Parameter horizon value must be >0.")

            # flexibilityMean
            if self.flexibilityMean is None:
                raise ValueError("Parameter flexibilityMean must be specified.")
            validate_value(self.flexibilityMean, lambda v: 0.0 <= v <= 1.0, "Parameter flexibilityMean must be in [0,1].")

            # flexibilityDev
            if self.flexibilityDev is None:
                raise ValueError("Parameter flexibilityDev must be specified.")
            validate_value(self.flexibilityDev, lambda v: 0.0 <= v <= 1.0, "Parameter flexibilityDev must be in [0,1].")

            # criticalTime
            if self.criticalTime is not None:
                validate_value(self.criticalTime, lambda v: 0 < v <= self.horizon, "Parameter criticalTime must be >0 and ≤ horizon.")
            
            # distributionPattern
            if self.distributionPattern is not None:
                dist = np.array(self.distributionPattern, dtype=float)
                if len(dist) != self.horizon:
                    raise ValueError("Parameter distributionPattern must be of same length as parameter horizon.")
                probaSum = 0
                for proba in dist:
                    probaSum+=proba
                if probaSum != 1:
                    raise ValueError("Probabilities of distributionPattern must sum to 1.")
                
            # preProcessingSSNDP
            if self.preProcessingSSNDP is None:
                raise ValueError("Parameter preProcessingSSNDP must be specified.")

    def __str__(self):
        return " | ".join(f"{k}={v}" for k, v in asdict(self).items())

class Network:
    def __init__(self, nodes, arcs, params = None, id = None, seed = None):
        self.nodes: list[Node] = nodes
        self.arcs: list[Arc] = arcs
        self.params: NetworkGeneratorParams = params
        self.id: int = id
        self.seed: int = seed

    @classmethod
    def from_file(cls, path: Path, isArcDistance: bool):
        nodes: list[Node] = []
        arcs: list[Arc] = []
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        if not lines[i].startswith("NODES,"):
            raise ValueError("Expected 'NODES,<count>' header at line 1")
        nodeNb = int(lines[i].split(",")[1])
        i += 1

        for _ in range(nodeNb):
            # id, clusterId (optional), x (optional), y (optional)
            splitLine = [p.strip() for p in lines[i].split(",")]
            i += 1

            clusterId = None
            x = y = None
            if len(splitLine) == 0:
                raise ValueError(f"Line {i}: node line missing")
            elif len(splitLine) == 2:
                val = splitLine[1]
                clusterId = int(val) if val.isdigit() and int(val) >= 0 else None
            elif len(splitLine) == 4:
                cId, sX, sY = splitLine[1:4]
                clusterId = int(cId) if cId.isdigit() and int(cId) >= 0 else None
                try:
                    x, y = float(sX), float(sY)
                except ValueError:
                    pass

            nodes.append(Node(x, y, clusterId))

        if not lines[i].startswith("ARCS,"):
            raise ValueError(f"Expected 'ARCS,<count>' header at line {i}")
        arcNb = int(lines[i].split(",")[1])
        i += 1

        for _ in range(arcNb):
            # id, src, dest, unit, fixed, capacity, distance
            splitLine = [p.strip() for p in lines[i].split(",")]
            i += 1

            if len(splitLine) == 0:
                raise ValueError(f"Line {i}: arc line missing")
            elif len(splitLine) >= 7:
                src, dest, unit, fixed, capacity, joker = splitLine[1:7]
                time, distance = (None, joker) if isArcDistance else (joker, None)
                arcs.append(Arc(int(src), int(dest), float(unit), float(fixed), float(capacity), none_float_option(distance), none_float_option(time)))
            else:
                raise ValueError(f"Line {i}: more parameters expected")

        return cls(nodes, arcs)
    
    def save(self, path: Path):
        """
        Save the network (nodes + arcs).
        """
        path.mkdir(parents=True, exist_ok=True)

        base_name = self.generate_file_name()
        out_path = path / base_name

        with open(out_path, "w") as f:
            f.write(f"NODES,{len(self.nodes)}\n")
            for i, node in enumerate(self.nodes):
                cId = "-1"
                if (node.clusterId is not None):
                    cId = node.clusterId
                x = "-1"
                y = "-1"
                if (node.x is not None):
                    x = node.x
                    y = node.y
                f.write(f"{i},{cId},{x},{y}\n")

            f.write(f"ARCS,{len(self.arcs)}\n")
            for i, arc in enumerate(self.arcs):
                f.write(
                    f"{i},{arc.src},{arc.dest},{arc.unit},{arc.fixed},"
                    f"{arc.capacity},{arc.distance}\n"
                )

    def generate_file_name(self):
        # Resolve density name depending on parameter specified.
        d=0
        if self.params.targetDensity is None:
            d=self.params.targetArcNb/(self.params.targetNodeNb*(self.params.targetNodeNb-1))
        else:
            d=self.params.targetDensity
        d=round(d*100)

        # Common part of file name.
        n=self.params.targetNodeNb
        fileName=f"_D{d}_N{n}_I{self.id}"
        if self.params.targetReciprocity is not None:
            r=round(self.params.targetReciprocity*100)
            fileName=f"_R{r}"+fileName

        # Resolve file name either for random or structured network.
        if self.params.randomGeneration:
            fileName="Random"+fileName
        else:
            dr=round(self.params.decayRate)
            a=round(self.params.hnRatio*100)
            uf=round(self.params.ufCostRatio*100)
            fileName=f"DR{dr}_A{a}_UF{uf}"+fileName
        
        if self.seed is not None:
            fileName += f"_S{self.seed}"
        return fileName + ".txt"

    def compute_metrics(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.nodes)))
        for arc in self.arcs:
            G.add_edge(arc.src, arc.dest, weight=arc.distance)

        return (nx.density(G), nx.reciprocity(G), nx.average_clustering(G), Network.get_avg_shortest_path_length(G), nx.degree_assortativity_coefficient(G))

    def get_avg_shortest_path_length(G: nx.DiGraph):
        node_list = list(G.nodes)
        distances = []

        for i in node_list:
            # Compute all shortest paths from i.
            lengths = nx.single_source_shortest_path_length(G, i)
            for j, dist in lengths.items():
                if i == j:
                    continue
                distances.append(dist)

        if len(distances) == 0:
            raise ValueError("The input network is disconnected.")

        avg_shortest_path_length = round(float(np.mean(distances)), 3)
        return avg_shortest_path_length

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
        if not lines[i].startswith("COMMODITIES,"):
            raise ValueError(f"Expected 'COMMODITIES,<count>' header at line {i}")
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

def validate_value(value, check_fn, error_msg):
    """
    Apply a validation function to either a single value or a list of values.
    Raises ValueError with the given message if any check fails.
    """
    if value is None:
        return
    if isinstance(value, list):
        for v in value:
            if not check_fn(v):
                raise ValueError(error_msg)
    else:
        if not check_fn(value):
            raise ValueError(error_msg)