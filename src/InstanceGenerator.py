import random
import math
import numpy as np
import scipy.stats as stats
from scipy.sparse.csgraph import dijkstra
from pathlib import Path

from Structures import Arc, Network, Commodity, InstanceGeneratorParams

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
            1. Convert arc distances into travel times.
            2. Compute all-pairs travel times.
            3. Identify valid origin–destination pairs and apply selection filters based on configuration parameters.
            4. Generate commodities with quantities and (for SSNDP) time windows.
            5. Optionally apply SSNDP preprocessing to use in solvers.

        Raises:
            ValueError: If fewer feasible commodities exist than requested.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Set arcs transit time.
        periodDuration = self.params.horizon * 24 / self.params.discretization # hours
        for arc in self.network.arcs:
            travelTime = arc.distance / self.params.speed # hours
            arc.time = math.ceil(travelTime / periodDuration) # number of periods

        # Generate quantity of all commodities.
        quantities = self.generate_commodities_quantities()

        # Compute travel times between all pairs, derive candidate commodity pairs in static network.
        travelTimes = self.compute_travel_times()
        validPairs = list(map(tuple, np.argwhere((travelTimes != np.inf) & (travelTimes != 0))))
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
            timedTuples = self.generate_timed_commodities(candidates, travelTimes)
            self.commodities.extend(Commodity(src, dest, quantity, avTime, dueTime) 
                                    for (src, dest, avTime, dueTime), quantity in zip(timedTuples, quantities))
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
            f.write(f"discretization={self.params.discretization}\n")
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
            d=str(self.params.discretization)
            fm=str(int(self.params.flexibilityMean*100))
            fd=str(int(self.params.flexibilityDev*100))
            fileName += f"_H{h}_D{d}_FM{fm}_FD{fd}"
            if self.params.criticalTime is not None:
                ct=str(int(self.params.criticalTime))
                fileName+=f"_CT{ct}"

        c=str(self.params.commodityNb)
        fileName += f"_C{c}_I{str(id)}_{basename}"
        return fileName

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
        finiteTimes = allPairTime[np.isfinite(allPairTime)]
        maxPath = np.max(finiteTimes)

        if maxPath >= self.params.discretization:
            raise Exception(
                f"Horizon not large enough for the size of the time-expanded network. "
                f"Longest path = {maxPath} periods; "
                f"Discretization = {self.params.discretization} periods."
            )

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
    
    def generate_timed_commodities(self, candidates, travelTimes):
        """
        Generate timed commodities for a SSNDP instance. While loop may take longer if
        the number of commodities asked is high compared to the max possible as it operates
        by draw and rejection.

        Args:
            candidates (list[tuple[int,int,int,int]]): All feasible origin destination pairs considered.
            travelTimes (np.ndarray): Matrix of travel times between origin destination pairs considered.

        Returns:
            list[tuple[int,int,int,int]]: Timed commodities generated (source, destination, available time, due time).
        """
        # Compute the parameters of the flexibility distribution.
        meanTravelTime = np.mean([travelTimes[src, dest] for src, dest in candidates])
        meanFlex = meanTravelTime * self.params.flexibilityMean
        stdDevFlex = meanFlex * self.params.flexibilityDev
        generatedSet = set()
        generated = []
        while len(generated) < self.params.commodityNb:
            # Draw an origin destination pair.
            src, dest = random.choice(candidates)

            # Draw a flexibility f.
            L = int(travelTimes[src, dest])
            maxFlex = self.params.discretization - 1 - L
            if maxFlex == 0:
                f = 0
            elif stdDevFlex == 0:
                f = min(int(round(meanFlex)), maxFlex)
            else:
                a = (0.0 - meanFlex) / stdDevFlex
                b = (maxFlex - meanFlex) / stdDevFlex
                sampledFlex = stats.truncnorm.rvs(a, b, loc=meanFlex, scale=stdDevFlex)
                f = int(math.ceil(sampledFlex))
                f = max(0, min(f, maxFlex))

            # Draw an available time e.
            maxAvailableTime = self.params.discretization - 1 - L - f
            if self.params.distributionPattern is None:
                e = random.randint(0, maxAvailableTime)
            else:
                probabilities = np.asarray(self.params.distributionPattern, dtype=float)
                feasibleProbabilities = probabilities[:maxAvailableTime + 1].copy()
                probabilitySum = feasibleProbabilities.sum()
                feasibleProbabilities /= probabilitySum
                e = int(np.random.choice(np.arange(maxAvailableTime + 1),p=feasibleProbabilities))

            # Derive a due time l based on the shortest path length L, available time e, flexibility f.
            l = e + L + f
            if l >= self.params.discretization:
                raise ValueError("Invalid due time value.")
            
            # Round down and up, respectively, the available and due time, based on critical time (if used).
            if (self.params.criticalTime is not None and self.params.criticalTime > 1):
                intervalLength = self.params.discretization / self.params.criticalTime
                e = int(math.floor(e / intervalLength) * intervalLength)
                l = int(math.ceil(l / intervalLength) * intervalLength)
                if l >= self.params.discretization:
                    continue

            # Check unicity of commodity and save.
            commodityTuple = (src, dest, e, l)
            if commodityTuple in generatedSet:
                continue
            generatedSet.add(commodityTuple)
            generated.append(commodityTuple)

        return generated