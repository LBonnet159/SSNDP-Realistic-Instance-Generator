import random
import math
import numpy as np
import scipy.stats as stats
from scipy.sparse.csgraph import dijkstra
from pathlib import Path

import matplotlib.pyplot as plt
from collections import Counter

from Structures import Arc, Network, Instance, Commodity, InstanceGeneratorParams

class InstanceGenerator:
    """
    Generate SNDP or SSNDP problem instances from a base network.

    Supports:
        - SNDP instance generation.
        - SSNDP instance generation.
        - Optional pre-processing for SSNDP time windows.
    """
    def __init__(self, seed, params: InstanceGeneratorParams, network: Network):
        """
        Initialize an instance generator.

        Args:
            seed (int): Random seed for reproducibility.
            params (InstanceGeneratorParams): Configuration parameters for instance generation.
            network (Network): Base network used to derive arcs, travel times, and clusters.
        """
        self.seed = seed
        self.params = params
        self.network = network
        self.timedArcs: list[Arc] = []
        self.commodities: list[Commodity] = []
        self.commoditySet = set()
        self.timeWindowsNode: list[list[tuple[int, int]]] = []
        self.timeWindowsArc: list[list[tuple[int, int]]] = []

    def generate(self):
        """
        Generate an SNDP or SSNDP instance based on the input network.

        Steps:
            1. Convert arc distances into travel times (normalized to the length of the planning horizon).
            2. Compute all-pairs travel times.
            3. Identify valid origin–destination pairs and apply selection filters based on configuration parameters.
            4. Generate commodities with quantities and (for SSNDP) time windows.
            5. Optionally apply SSNDP preprocessing to use in solvers.

        Raises:
            ValueError: If fewer feasible commodities exist than requested.
        """
        # Find longest arc.
        longestArcDist = 0.0
        for arc in self.network.arcs:
            if arc.distance > longestArcDist:
                longestArcDist = arc.distance

        # Set longest (in distance) arc to take 0.2 of the horizon (in time). Apply to all arcs. 
        convRate = (self.params.horizon*0.2)/longestArcDist
        for arc in self.network.arcs:
            arc.time = math.ceil(arc.distance*convRate)

        # Generate quantity of all commodities.
        quantities = self.generate_commodities_quantities()

        # Compute travel times between all pairs, derive candidate commodity pairs in static network.
        travelTimes = self.compute_travel_times()
        validPairs = list(map(tuple, np.argwhere((travelTimes != np.inf) & (travelTimes != 0))))
        meanTravelTime = 0.0
        for pair in validPairs:
            meanTravelTime += travelTimes[pair[0],pair[1]]
        meanTravelTime /= len(validPairs)
        random.shuffle(validPairs)
        
        # Filter candidates according to user options. 
        candidates = self.select_valid_pairs(validPairs)

        # Generate commodities.
        if self.params.doStatic:
            if len(candidates) < self.params.commodityNb:
                raise ValueError(f"Commodity number asked ({self.params.commodityNb}) larger than maximum number possible ({len(candidates)}).")
            staticPairs = random.sample(candidates, self.params.commodityNb)
            self.commodities.extend(Commodity(src, dest, quantity) for (src, dest), quantity in zip(staticPairs, quantities))
        else:
            flexibleTimes = self.generate_flexible_times(meanTravelTime)
            tuples = self.generate_timed_commodities(candidates, travelTimes, flexibleTimes)
            self.commodities.extend(Commodity(src, dest, quantity, avTime, dueTime) 
                                    for (src, dest, avTime, dueTime), quantity in zip(tuples, quantities))
            if self.params.preProcessingSSNDP:
                self.generate_preprocessings()

        if len(self.commodities) < self.params.commodityNb:
            print(f"Commodity number generated ({len(self.commodities)}) lower than commodity number asked ({self.params.commodityNb}).")

    def save(self, path, folder, basename: str, id: int):
        """
        Save the generated instance in a .txt format.

        Args:
            path (str): Output directory base path.
            folder (str | None): Optional subfolder name.
            basename (str): Base name for the file.
            id (int): Instance ID, appended to filename.

        Notes:
            - Saves nodes, arcs, and commodities (with time data if SSNDP).
            - Includes optional node/arc time windows if pre-processing is active.
            - File format is compatible with other SSNDP tools in this project.
        """
        if folder is not None:
            path += "/" + folder
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        outPath = path / self.generate_file_name(basename, id)
        f = open(outPath,"w")
        f.write(f"NODES,{len(self.network.nodes)}\n")
        i = 0
        for node in self.network.nodes:
            line=f"{i}"
            if node.clusterId is not None: line+=f",{node.clusterId}"
            if node.x is not None and node.y is not None: line+=f",{node.x},{node.y}"
            f.write(line+"\n")
            i+=1
        f.write(f"ARCS,{len(self.network.arcs)}\n")
        i = 0
        for arc in self.network.arcs:
            f.write(f"{i},{arc.src},{arc.dest},{arc.unit},{arc.fixed},{arc.capacity}")
            if not self.params.doStatic:
                f.write(f",{arc.time}")
            f.write("\n")
            i+=1
        f.write(f"COMMODITIES,{len(self.commodities)}\n")
        i = 0
        for commodity in self.commodities:
            f.write(f"{i},{commodity.origin},{commodity.destination},{commodity.quantity}")
            if not self.params.doStatic:
                f.write(f",{commodity.availableTime},{commodity.dueTime}")
            f.write("\n")
            i += 1
        if not self.params.doStatic:
            f.write(f"horizon={self.params.horizon}\n")
        if self.params.distributionPattern is not None:
            f.write(f"distribution_pattern={self.params.distributionPattern}\n")
        if len(self.timeWindowsNode) > 0:
            length = sum(1 for inner in self.timeWindowsNode for item in inner if item is not None)
            f.write(f"COMMODITY_NODE_TIMEWINDOWS,{length}\n")
            i=0
            for outerIdx, outerList in enumerate(self.timeWindowsNode):
                for innerIdx, tw in enumerate(outerList):
                    if tw is not None:
                        # twId, commodityId, nodeId, lb, ub
                        lb, ub = tw  # unpack the tuple
                        f.write(f"{i},{outerIdx},{innerIdx},{lb},{ub}\n")
                        i += 1
        if len(self.timeWindowsArc) > 0:
            length = sum(1 for inner in self.timeWindowsArc for item in inner if item is not None)
            f.write(f"COMMODITY_ARC_TIMEWINDOWS,{length}\n")
            i=0
            for outerIdx, outerList in enumerate(self.timeWindowsArc):
                for innerIdx, tw in enumerate(outerList):
                    if tw is not None:
                        # twId, commodityId, arcId, lb, ub
                        lb, ub = tw  # unpack the tuple
                        f.write(f"{i},{outerIdx},{innerIdx},{lb},{ub}\n")
                        i += 1
        f.close()

    def generate_file_name(self, basename: str, id: int):
        """
        Generate a descriptive filename encoding key parameters.

        Args:
            basename (str): Base string for the filename.
            id (int): Instance identifier.

        Returns:
            str: A filename summarizing generation parameters (ratios, horizon...).
        """
        fileName = ""
        if not self.params.doStatic:
            fileName+="S"

        mcq=str(int(self.params.quantityToCapaMean*100))
        dcq=str(int(self.params.quantityToCapaDev*100))
        fileName+=f"SNDP_MCQ{mcq}_DCQ{dcq}"
        if self.params.sameRegionRatio is not None:
            sr=str(int(self.params.sameRegionRatio*100))
            fileName+=f"_SR{sr}"
        if self.params.disparityRatio is not None:
            dr=str(int(self.params.disparityRatio*100))
            fileName+=f"_DR{dr}"        

        if not self.params.doStatic:
            h=str(self.params.horizon)
            fm=str(int(self.params.flexibilityMean*100))
            fd=str(int(self.params.flexibilityDev*100))
            fileName += f"_H{h}_FM{fm}_FD{fd}"
            if self.params.criticalTime is not None:
                ct=str(int(self.params.criticalTime))
                fileName+=f"_CT{ct}"

        c=str(self.params.commodityNb)
        fileName += f"_C{c}_I{str(id)}_{basename}"
        return fileName

    def enumerate_timed_commodities(self, validPairs, travelTimes, flexibleTime):
        """
        Enumerate all feasible timed commodities for SSNDP.

        For each valid pair, generate all (available, due) time combinations
        within the horizon, applying flexibility and critical-time rounding
        if the options are activated.

        Args:
            validPairs (list[tuple[int,int]]): Valid origin–destination pairs.
            travelTimes (np.ndarray): Matrix of travel times.
            flexibleTime (np.ndarray): Per-commodity flexibility durations.

        Returns:
            list[tuple[int,int,int,int]]: Feasible (src, dest, available, due) tuples.

        Raises:
            ValueError: If unfeasible timing combinations are produced.
        """
        timedCandidates = []
        i = 0
        for src, dest in validPairs:
            L = int(travelTimes[src][dest])
            f = flexibleTime[i]
            maxStart = self.params.horizon - 1 - f - L
            if maxStart < 0:
                raise ValueError("Unfeasible timed commodity.")
                
            # Enumerate all possible available times
            for e in range(maxStart + 1):
                l = e + L + f
                if l >= self.params.horizon:
                    continue
                
                if self.params.criticalTime is not None and self.params.criticalTime > 1:
                    e = math.floor(e / self.params.criticalTime) * self.params.criticalTime
                    l = math.ceil(l / self.params.criticalTime) * self.params.criticalTime
                    if l >= self.params.horizon:
                        continue
                
                timedCandidates.append((src, dest, e, l))
        
        return timedCandidates

    def generate_timed_commodities(self, validPairs, travelTimes, flexibleTime):
        """
        Select a subset of feasible timed commodities.

        Args:
            validPairs (list[tuple[int,int]]): Valid origin destination pairs.
            travelTimes (np.ndarray): Travel times matrix.
            flexibleTime (np.ndarray): Array of flexibility times.

        Returns:
            list[tuple[int,int,int,int]]: Selected timed commodities.

        Raises:
            ValueError: If not enough feasible timed commodities exist.
        """

        # Enumerate all possible timed commodities.
        candidates = self.enumerate_timed_commodities(validPairs, travelTimes, flexibleTime)
        if len(candidates) < self.params.commodityNb:
            raise ValueError(f"Only {len(candidates)} feasible commodities exist, less than requested {self.params.commodityNb}.")
        
        # Select candidates according to user specified options.
        selected = []
        if self.distribution_pattern is not None:
            selected = self.distribution_pattern(candidates)
        else:
            selected = random.sample(candidates, self.params.commodityNb)
        return selected

    def generate_flexible_times(self, meanTravelTime):
        meanFlex=meanTravelTime*self.params.flexibilityMean
        stdDevFlex=meanFlex*self.params.flexibilityDev
        flexibleTimes = np.random.normal(loc=meanFlex, scale=stdDevFlex, size=self.params.commodityNb)
        flexibleTimes = np.ceil(np.maximum(flexibleTimes, 0)).astype(int) # Set value below 0 to 0.
        return flexibleTimes

    def generate_commodities_quantities(self):
        """
        Generate commodity quantities from a truncated normal distribution.

        Quantities are drawn between 1% and 100% of the smallest arc capacity,
        scaled by `quantityToCapaMean` and `quantityToCapaDev`.

        Returns:
            np.ndarray: Array of generated commodity quantities.
        """
        minCapacity = min([arc.capacity for arc in self.network.arcs])
        meanQuantity = minCapacity * self.params.quantityToCapaMean
        stdDevQuantity = meanQuantity * self.params.quantityToCapaDev
        minQuantity = 0.01*minCapacity
        truncArg1 = (minQuantity - meanQuantity) / stdDevQuantity
        truncArg2 = (minCapacity - meanQuantity) / stdDevQuantity
        quantity = stats.truncnorm.rvs(truncArg1, truncArg2, loc=meanQuantity, scale=stdDevQuantity, size=self.params.commodityNb)
        return np.round(quantity, decimals=5)

    def compute_travel_times(self):
        """
        Compute all-pairs travel times using Dijkstra’s algorithm.

        Returns:
            np.ndarray: Matrix of shortest travel times between all node pairs.

        Raises:
            Exception: If the configured time horizon is too small for the network.
        """
        allPairTime = InstanceGenerator.compute_all_pair_time(len(self.network.nodes), self.network.arcs)

        # Check horizon is valid.
        maxFiniteDistances=[]
        for item in allPairTime.tolist():
            maxFiniteDistances.append(max(d for d in item if d != np.inf))
        maxPath=max(d for d in maxFiniteDistances if d != np.inf)
        if (maxPath>=self.params.horizon):
            raise Exception(f"Horizon not large enough for the size of the time-expanded network." \
            "Longest path = {maxPath} ; Horizon = {horizon}. Reduce network or enlarge horizon.")
        
        return allPairTime
    
    @staticmethod
    def compute_all_pair_time(nodeNb, arcs: list[Arc]):
        """
        Compute all-pairs travel time matrix from arc list.

        Args:
            nodeNb (int): Number of nodes.
            arcs (list[Arc]): List of directed arcs with `time` attribute.

        Returns:
            np.ndarray: Directed shortest-path matrix (∞ if unreachable).
        """
        arcNb = len(arcs)

        adjMatrix = np.full((nodeNb, nodeNb), np.inf)
        for i in range(arcNb):
            adjMatrix[arcs[i].src][arcs[i].dest] = arcs[i].time

        return dijkstra(csgraph=adjMatrix, directed=True)

    def generate_preprocessings(self):
        """
        Preprocess an SSNDP instance to compute node and arc time windows.

        For each commodity:
            - Compute feasible time windows at each node (earliest arrival, latest departure).
            - Derive arc time windows from node windows and arc travel times.

        Notes:
            - Time windows with lb > ub are discarded.
            - Results are stored in `timeWindowsNode` and `timeWindowsArc`.
        """
        nodeNb = len(self.network.nodes)
        arcNb = len(self.network.arcs)
        commodityNb = len(self.commodities)

        # We compute the transit time of all node pairs.
        allPairTime = InstanceGenerator.compute_all_pair_time(nodeNb, self.network.arcs)

        # We compute the time windows of each commodity to each node in the static network.
        self.timeWindowsNode = [[None for _ in range(nodeNb)] for _ in range(commodityNb)]
        for i in range(commodityNb):
            currentK = self.commodities[i]
            for j in range(nodeNb):
                lb = int(currentK.availableTime + allPairTime[currentK.origin, j])
                ub = int(currentK.dueTime - allPairTime[j, currentK.destination])
                if lb > ub:
                    continue # Empty node time window.

                self.timeWindowsNode[i][j] = (lb,ub)

        # We compute the time windows of each commodity to each arc in the static network.
        self.timeWindowsArc = [[None for _ in range(arcNb)] for _ in range(commodityNb)]
        for i in range(commodityNb):
            currentK = self.commodities[i]
            for j in range(arcNb):
                currentArc = self.network.arcs[j]
                if self.timeWindowsNode[i][currentArc.src] is None or self.timeWindowsNode[i][currentArc.dest] is None:
                    continue # Empty source or destination node time window.

                lb = self.timeWindowsNode[i][currentArc.src][0]
                ub = self.timeWindowsNode[i][currentArc.dest][1] - currentArc.time
                if lb > ub:
                    continue # Empty arc time window.

                self.timeWindowsArc[i][j] = (lb,ub)

    def select_valid_pairs(self, validPairs):
        """
        Apply selection filters on origin destination pairs.

        Filters include:
            - Disparity ratio
            - Same-region ratio

        Args:
            validPairs (list[tuple[int,int]]): All valid origin destination pairs.

        Returns:
            list[tuple[int,int]]: Filtered and shuffled valid origin destination pairs.
        """

        # Start with full set
        selectedPairs = validPairs.copy()

        # Apply disparity ratio if activated
        if self.params.disparityRatio is not None:
            selectedPairs = self.disparity_ratio_generation(selectedPairs)
            if len(selectedPairs) < self.params.commodityNb and self.params.doStatic:
                print(f"Not enough candidate commodities possible ({len(selectedPairs)}) with the specified disparity ratio ({self.params.disparityRatio}).")

        # Apply same region ratio if activated
        if self.params.sameRegionRatio is not None:
            samePairs, diffPairs = self.same_region_generation(selectedPairs)
            sameNb = math.floor(self.params.sameRegionRatio * len(selectedPairs))
            diffNb = len(selectedPairs) - sameNb
            selectedPairs = random.sample(samePairs, min(sameNb, len(samePairs))) + \
                            random.sample(diffPairs, min(diffNb, len(diffPairs)))
            if len(selectedPairs) < self.params.commodityNb and self.params.doStatic:
                print(f"Not enough candidate commodities possible ({len(selectedPairs)}) with the specified same region ratio ({self.params.sameRegionRatio}).")

        random.shuffle(selectedPairs)
        return selectedPairs

    def same_region_generation(self, validPairs):
        """
        Split valid origin destination pairs into same cluster and distinct cluster sets.

        Args:
            validPairs (list[tuple[int,int]]): Valid origin destination pairs.

        Returns:
            tuple[list, list]: (sameRegionPairs, diffRegionPairs)
        """
        sameRegionPairs = []
        diffRegionPairs = []
        for src, dest in validPairs:
            if self.network.nodes[src].clusterId == self.network.nodes[dest].clusterId:
                sameRegionPairs.append((src, dest))
            else:
                diffRegionPairs.append((src, dest))
        return sameRegionPairs, diffRegionPairs

    def disparity_ratio_generation(self, validPairs):
        """
        Apply disparity ratio to skew commodity distribution across clusters.

        Implements a discrete power-law distribution to control
        how origins/destinations are biased toward specific clusters.

        Args:
            validPairs (list[tuple[int,int]]): Candidate origin destination pairs.

        Returns:
            list[tuple[int,int]]: Filtered list of origin destination pairs following the ratio.

        Notes:
            Prints a warning if quotas cannot be met with available candidates.
        """
        # Compute the number of clusters.
        clusters = list({self.network.nodes[i].clusterId for pair in validPairs for i in pair})
        clusterNb = len(clusters)

        # Draw discteized power law.
        sMin = 0
        sMax = 2
        s = sMin + (sMax-sMin)*self.params.disparityRatio
        ranks = np.arange(1, clusterNb + 1)
        weights = 1 / (ranks ** s)
        probs = weights / weights.sum()
        random.shuffle(clusters)
        clusterProb = {cid: p for cid, p in zip(clusters, probs)}

        total = len(validPairs)
        clusterQuota = {cid: total * clusterProb[cid] for cid in clusters}
        clusterOriginQuota = {cid: quota / 2 for cid, quota in clusterQuota.items()}
        clusterDestQuota = {cid: quota / 2 for cid, quota in clusterQuota.items()}

        selected = []
        for i, j in validPairs:
            ci, cj = self.network.nodes[i].clusterId, self.network.nodes[j].clusterId
            if clusterOriginQuota[ci] > 0 and clusterDestQuota[cj] > 0:
                selected.append((i, j))
                clusterOriginQuota[ci] -= 1
                clusterDestQuota[cj] -= 1

        fail = False
        for quota in clusterOriginQuota:
            if quota > 0:
                fail = True
        for quota in clusterDestQuota:
            if quota > 0:
                fail = True
        if fail:
            print("The distribution of commodities origin and destination according to disparity ratio failed. " \
            "Not enough candidates available.")

        return selected
    
    def distribution_pattern(self, candidates):
        """
        Select timed commodities according to a temporal distribution pattern.

        Args:
            candidates (list[tuple[int,int,int,int]]): All feasible timed commodities.

        Returns:
            list[tuple[int,int,int,int]]: Selected commodities matching the distribution pattern.

        Notes:
            - Uses `params.distributionPattern` as a probability vector over time slots.
            - If insufficient candidates exist in some slots, random pairs from other time slots are used as a fallback.
        """
        dist = np.array(self.params.distributionPattern, dtype=float)

        # Group candidates by available time.
        groupTime = {t: [] for t in range(self.params.horizon)}
        for tup in candidates:
            e = tup[2]
            if e < self.params.horizon:
                groupTime[e].append(tup)

        # Compute quotas per time slot
        targetCounts = np.round(dist * self.params.commodityNb).astype(int)
        selected = []

        # First pass: sample within each time slot up to its quota
        for t in range(self.params.horizon):
            pool = groupTime.get(t, [])
            quota = targetCounts[t]
            if len(pool) == 0 or quota == 0:
                continue
            chosen = random.sample(pool, min(quota, len(pool)))
            selected.extend(chosen)

        # If still missing commodities, fill from remaining candidates randomly
        deficit = self.params.commodityNb - len(selected)
        if deficit > 0:
            remaining = [c for c in candidates if c not in selected]
            if len(remaining) < deficit:
                print(f"Warning: only {len(remaining)} remaining candidates to fill {deficit} slots.")
                deficit = len(remaining)
            selected.extend(random.sample(remaining, deficit))

        return selected