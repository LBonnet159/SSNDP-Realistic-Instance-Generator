import math
import random
import statistics
import time
import numpy as np
import heapq
import matplotlib.pyplot as plt
from itertools import permutations
from pathlib import Path
from Structures import Node, Arc, NetworkGeneratorParams, Network

class NetworkGenerator:
    """
    Generates hub-and-spoke or random networks with specified structural properties.

    Supports:
        - Purely random network generation.
        - Structured hub-and-spoke generation.
        - Network emulation based on an input reference network.
    """
    def __init__(self, seed: int | None, params: NetworkGeneratorParams):
        """
        Initialize the network generator.

        Args:
            - seed (int | None): Seed for reproducibility. If None, uses non-deterministic randomness.
            - params (NetworkGeneratorParams): Configuration parameters controlling generation.
        """
        self.seed = seed
        self.params = params
        self.rng = random.Random(0)
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.nodes: list[Node] = []
        self.arcs: list[Arc] = []
        self.pairs = set()
        self.adjustedR = 0.0
        self.hubsId: list[int] = []
        self.clusteredNodesId = []
        self.targetArcNb = 0

        self.id = 0 # Number of networks generated.

    def emulate_network(self, networkNb):
        """
        Generate hub-and-spoke networks that emulate the structure of an existing network.

        The method reads a reference network specified in the configuration file and generates
        a pool of candidate networks that have the following similar characteristics: node number,
        density, reciprocity.
        It then selects the `networkNb` networks with the smallest distance from 
        the reference network in terms of assortativity, clustering coefficient, 
        and average shortest path length.

        Args:
            networkNb (int): Number of best networks to retain after emulation.

        Returns:
            list[Network]: The `networkNb` generated networks most similar to the reference.

        Notes:
            - Uses normalized absolute distance on selected metrics for similarity scoring.
            - If the reference network has coordinates, its bounding box defines spatial limits.
        """
        if (self.params.networkEmulationPath is None):
            raise ValueError("Path to network to emulate not specified. Specify in Config.txt file.")
        if (self.params.networkEmulationTimeLimit is None):
            raise ValueError("Time limit for network emulation not specified. Specify in Config.txt file.")
        
        # Set seed for reproductibility.
        networkSeed = self.rng.randint(0, 2**32-1)
        np.random.seed(networkSeed)
        random.seed(networkSeed)

        # Read network to emulate.
        refNetwork = Network.from_file(Path(self.params.networkEmulationPath), isArcDistance=True)
        
        # Set bounding box.
        refBboxWidth = refBboxHeight = 0
        if (refNetwork.nodes[0].x is not None):
            # If node coordinates in reference network, determine bounding box.
            refMaxHeight=0
            refMaxWidth=0
            if refNetwork.nodes[0].x is not None:
                for node in refNetwork.nodes:
                    if node.x > refMaxWidth:
                        refMaxWidth = node.x
                    if node.y > refMaxHeight:
                        refMaxHeight = node.y

            refBboxHeight = refMaxHeight*0.05
            refBboxWidth = refMaxWidth*0.05
        elif (self.params.bboxHeight is not None):
            # Else, if specified, use bounding box from config file.
            refBboxHeight = self.params.bboxHeight
            refBboxWidth = self.params.bboxWidth
        else:
            # Else, use a default [0,100] bounding box.
            refBboxHeight = 100
            refBboxWidth = 100

        # Compute the mean capacity and unit to fixed cost ratio.
        capacities=[]
        priceRatio=[]
        for arc in refNetwork.arcs:
            capacities.append(arc.capacity)
            priceRatio.append(arc.unit / arc.fixed)
        refAvgCapa = statistics.mean(capacities)
        refAvgPriceRatio = statistics.mean(priceRatio)
        
        # Compute CNA metrics of reference graph.
        refDensity, refRecipro, refAvgCC, refAvgSPL, refASR = refNetwork.compute_metrics()

        # Set generator for emulation.
        bestNetworks = []
        timeLimit = self.params.networkEmulationTimeLimit
        start = time.time()
        genParams = NetworkGeneratorParams(None, None, False, refAvgCapa, refBboxWidth, refBboxHeight, 
                                           self.params.targetNodeNb, None, refDensity, refRecipro, 30, self.params.hnRatio, refAvgPriceRatio)
        generator = NetworkGenerator(self.seed, genParams)

        minScore = float("inf")
        maxScore = float("-inf")
        totalGenerated = 0
        while time.time()-start < timeLimit:
            # Draw randomly hub node ratio if not specified.
            if (self.params.hnRatio is None):
                generator.params.hnRatio = random.uniform(0.0,0.3)

            # Generate network.
            genNetwork = generator.generate()
            genDensity, genReciprocity, genAvgCC, genAvgSPL, genASR = genNetwork.compute_metrics()

            # Computes distances and score.
            drDistance = math.sqrt(((genDensity - refDensity) / refDensity) ** 2 + ((genReciprocity - refRecipro) / refRecipro) ** 2)
            avgCCavgSPLDistance = math.sqrt(((genAvgCC - refAvgCC) / refAvgCC) ** 2 + ((genAvgSPL - refAvgSPL) / refAvgSPL) ** 2)
            asrDistance = abs((genASR - refASR) / refASR)
            score = drDistance + avgCCavgSPLDistance + asrDistance

            totalGenerated += 1
            minScore = min(minScore, score)
            maxScore = max(maxScore, score)

            heapq.heappush(bestNetworks, (-score, genNetwork))
            if len(bestNetworks) > networkNb:
                heapq.heappop(bestNetworks)

        # Order by score and keep the best networkNb networks
        bestNetworksSorted = sorted(bestNetworks, key=lambda t: t[0], reverse=True)
        listNetworks = [t[1] for t in bestNetworksSorted]
        
        # Print statistics.
        print(f"Network emulation of file: {self.params.networkEmulationPath}")
        print(f"Number of networks tested: {totalGenerated}")
        print(f"Minimum representativity score: {minScore:.4f}")
        print(f"Maximum representativity score: {maxScore:.4f}")

        return listNetworks

    def generate(self, id=None):
        """
        Generate a single network according to configured parameters.

        Depending on settings, produces either a random or structured
        hub-and-spoke network and ensures weak connectivity.

        Args:
            id (int, optional): Identifier to reproduce a specific random instance.

        Returns:
            Network: The generated directed network.

        Notes:
            - Automatically computes target arc count and reciprocity if not specified.
            - Can optionally visualize the generated layout for debugging.
        """
        # Set seed for reproductibility.
        networkSeed = self.rng.randint(0, 2**32-1)
        if (id is not None):
            for i in range(id-1): # Handle reproductibility of a specific network.
                networkSeed = self.rng.randint(0, 2**32-1)
        np.random.seed(networkSeed)
        random.seed(networkSeed)

        self.compute_arc_number()

        # Fix reciprocity at target density if not specified.
        if self.params.targetReciprocity is None:
            self.params.targetReciprocity = self.targetArcNb/(self.params.targetNodeNb*(self.params.targetNodeNb-1))

        if self.params.randomGeneration:
            self.random_generation()
        else:
            self.node_generation()
            self.arc_generation()

        if __debug__:
            self.validity_check()

            # Plot network for debugging purposes.
            plotNetwork=False
            if plotNetwork:
                self.plot_network()

        self.arcs = self.arcs[:self.targetArcNb]
        network = Network(self.nodes[:], self.arcs[:], self.params, self.id, self.seed)
        self.id += 1
        self.clear()
        return network

    def random_generation(self):
        """
        Generate a purely random directed network.

        Nodes are placed uniformly in [0,1] x [0,1].
        Arcs are created randomly until the target arc count is reached,
        optionally enforcing reciprocity.

        Notes:
            Uses `targetReciprocity` if specified.
        """

        useReciprocity = self.params.targetReciprocity is not None
        if useReciprocity:
            self.adjustedR = self.params.targetReciprocity/(2-self.params.targetReciprocity)
        
        random_points = np.random.rand(self.params.targetNodeNb, 2)
        self.nodes.extend(Node(x, y, 0) for x, y in random_points)
        allPairs = list(permutations(range(self.params.targetNodeNb),2))
        random.shuffle(allPairs)
        for i, j in allPairs:
            if ((i, j) not in self.pairs) and ((j, i) not in self.pairs):
                newArcs = self.reciprocity_arc_generation(useReciprocity, i, j)
                self.update_arc_pairs(newArcs)
                if len(self.arcs) >= self.targetArcNb:
                    break

    def node_generation(self):
        """
        Generate clustered nodes for a hub-and-spoke network.

        Each cluster corresponds to one hub and its spokes.
        Node coordinates are drawn around cluster centers using an exponential decay.

        Notes:
            - The number of clusters equals ceil(hnRatio * targetNodeNb).
            - All nodes are bounded within [0,1] x [0,1].
        """
         
        # We set the number of clusters.
        clusterNb = math.ceil(self.params.hnRatio * self.params.targetNodeNb)
        if clusterNb == 0:
            clusterNb = 1

        # We generate the center of each cluster, in a [0, 1] x [0, 1] plane.
        clusterCenters = np.random.rand(clusterNb, 2)
        
        # We assign the points to each cluster.
        remainingPointsNb = self.params.targetNodeNb - clusterNb
        pointsPerCluster = [remainingPointsNb // clusterNb] * clusterNb
        for i in range(remainingPointsNb % clusterNb):
            # Any remaining point is randomly assigned to a cluster.
            pointsPerCluster[random.randint(0,clusterNb-1)] += 1

        # We generate the hub and spoke points.
        for i in range(clusterNb):
            clusterCenter = clusterCenters[i]
            clusterSize = pointsPerCluster[i]
            nodesId = []
            for _ in range(clusterSize):
                x, y = self.generate_cluster_node(clusterCenter)
                while not (0 <= x <= 1 and 0 <= y <= 1):
                    # Points must be in the [0, 1] x [0, 1] range.
                    x, y = self.generate_cluster_node(clusterCenter)

                nodesId.append(len(self.nodes))
                x = round(x * self.params.bboxWidth,5)
                y = round(y * self.params.bboxHeight, 5)
                self.nodes.append(Node(x,y,i))
                
            nodesId.append(len(self.nodes))
            self.hubsId.append(len(self.nodes))
            x = round(clusterCenter[0] * self.params.bboxWidth, 5)
            y = round(clusterCenter[1] * self.params.bboxHeight, 5)
            self.nodes.append(Node(x,y,i))
            self.clusteredNodesId.append(nodesId)

    def arc_generation(self):
        """
        Generate arcs for a hub-and-spoke network.

        Steps:
            1. Connect hubs to form a backbone.
            2. Connect spokes to their cluster hub.
            3. Add intra-cluster (spoke–spoke) arcs.
            4. Add inter-cluster (cross-cluster) arcs.
            5. Fill remaining arc budget randomly.

        Raises:
            Exception: If arc budget too low for weak connectivity.
        """
        nodeNb = len(self.nodes)
        if (nodeNb != self.params.targetNodeNb):
            raise Exception(f"Node number produced not correct: asked={self.params.targetNodeNb} ; generated={nodeNb}")

        clusterNb = len(self.clusteredNodesId)
        self.adjustedR = self.params.targetReciprocity/(2-self.params.targetReciprocity)
        self.adjustedR = max(0,self.adjustedR)
        
        # We compute if the arc budget is enough to produce a weakly connected graph.
        minBackboneArcNb = 2*(clusterNb-1)
        minSpokeHubArcNb = (nodeNb-clusterNb)*(self.adjustedR+1)
        if (minBackboneArcNb+minSpokeHubArcNb > self.targetArcNb):
            raise Exception(f"Arc budget too low. Increase node number or density, or lower the reciprocity or the hub and spokes ratio. " \
            f"Budget={self.targetArcNb} ; Minimum arc needed={minBackboneArcNb+minSpokeHubArcNb}")

        # Step 1: We generate reciprocal arcs between hubs.
        self.generate_hub_hub_arcs()
        arcBudget = self.targetArcNb - len(self.arcs)
        self.adjustedR = (self.targetArcNb*self.adjustedR-len(self.arcs))/arcBudget

        # Step 2: We generate arcs between spokes and hubs.
        self.generate_same_cluster_hub_spoke_arcs()

        # Step 3: We generate intra cluster arcs, i.e., arcs between spokes of the same cluster.
        arcBudget = self.targetArcNb - len(self.arcs)
        if (arcBudget > 0):
            intraArcNb = self.compute_intra_arc_nb(arcBudget)
            self.generate_same_cluster_spoke_spoke_arcs(intraArcNb)

        # Step 4: We generate inter cluster arcs, i.e., arcs between spokes not of the same cluster.
        arcBudget = self.targetArcNb - len(self.arcs)
        if (arcBudget > 0):
            interArcNb = self.compute_inter_arc_nb(arcBudget)
            self.generate_distinct_cluster_spoke_spoke_arcs(interArcNb)

        # Step 5: We randomly generate the remaining arcs.
        arcBudget = self.targetArcNb - len(self.arcs)
        if (arcBudget > 0):
            self.generate_distinct_cluster_hub_spoke_arcs(arcBudget)

    def generate_hub_hub_arcs(self):
        """
        Generate reciprocal backbone arcs between hubs.
        """
        # If generating a complete network of hub nodes isn't possible, 
        # we find the maximum number of neighbors per hub possible.
        hubNb = len(self.hubsId)
        maxBackboneArcNb = hubNb*(hubNb-1)
        spokeNb = len(self.nodes)-hubNb
        maxSpokeHubArcNb = math.ceil(spokeNb*(self.params.targetReciprocity+1))
        neighborNb = math.floor(hubNb/2)
        if maxBackboneArcNb+maxSpokeHubArcNb > self.targetArcNb:
            neighborNb -= 1
            while neighborNb >= 2:
                newBackboneArcNb = hubNb*neighborNb
                if newBackboneArcNb + maxSpokeHubArcNb <= self.targetArcNb:
                    break
                neighborNb -= 1

        if neighborNb >= len(self.hubsId):
            raise Exception(f"Hub asked to be linked to more hubs ({neighborNb}) than possible ({len(self.hubsId)}).")
        
        # We connect the hubs together reciprocically in a round-robin fashion.
        for i in range(len(self.hubsId)):
            for j in range(1, neighborNb+1):
                u, v = self.hubsId[i], self.hubsId[(i+j) % len(self.hubsId)]
                if (u,v) in self.pairs:
                    continue
                newArcs = self.reciprocity_arc_generation(False, u, v)
                self.update_arc_pairs(newArcs)

    def generate_same_cluster_hub_spoke_arcs(self):
        """
        Generate arcs between each hub and its associated spokes.
        """
        # First pass, generate an arc randomly between each spoke and its related hub.
        for i in range(len(self.hubsId)):
            for spokeId in self.clusteredNodesId[i][:-1]:
                if random.random() < 0.5:
                    self.update_arc_pairs([self.get_new_arc(self.hubsId[i], spokeId)])
                else:
                    self.update_arc_pairs([self.get_new_arc(spokeId, self.hubsId[i])])

        # Second pass, until budget reach or all pairs considered, generate arcs respecting the reciprocity.
        for i in range(len(self.hubsId)):
            for spokeId in self.clusteredNodesId[i][:-1]:
                if random.random() < self.adjustedR:
                    if (self.hubsId[i],spokeId) in self.pairs:
                        self.update_arc_pairs([self.get_new_arc(spokeId, self.hubsId[i])])
                else:
                    if (spokeId,self.hubsId[i]) in self.pairs:
                        self.update_arc_pairs([self.get_new_arc(self.hubsId[i], spokeId)])
                if len(self.arcs) >= self.targetArcNb:
                    break
            if len(self.arcs) >= self.targetArcNb:
                    break

    def compute_intra_arc_nb(self, arcBudget):
        """
        Compute number of intra-cluster arcs based on remaining arc budget.
        """
        clusterNb = len(self.hubsId)

        # Compute max possible number of intra arcs to generate.
        maxIntraArcNb = 0
        for i in range(clusterNb):
            iSpokeNb = len(self.clusteredNodesId[i])-1
            maxIntraArcNb += iSpokeNb * (iSpokeNb-1)
        
        result = min(arcBudget, maxIntraArcNb)
        return result

    def compute_inter_arc_nb(self, arcBudget):
        """
        Compute number of inter-cluster arcs based on remaining arc budget.
        """
        clusterNb = len(self.hubsId)

        # Compute max possible number of intra and inter arcs to generate.
        maxInterArcNb = 0
        for i in range(clusterNb-1):
            iSpokeNb = len(self.clusteredNodesId[i])-1
            for j in range(i+1, clusterNb):
                jSpokeNb = len(self.clusteredNodesId[j])-1
                maxInterArcNb += iSpokeNb * jSpokeNb * 2

        result = min(arcBudget, maxInterArcNb)
        return result
    
    def generate_same_cluster_spoke_spoke_arcs(self, targetIntraArcNb):
        """
        Generate arcs between spokes within the same cluster.
        """
        if targetIntraArcNb == 0:
            return
        
        pairs=[]
        for i in range(len(self.clusteredNodesId)):
            spokes = self.clusteredNodesId[i][:-1]
            for j in range(len(spokes)-1):
                for k in range(j+1,len(spokes)):
                    pairs.append((spokes[j],spokes[k]))
        random.shuffle(pairs)
        self.generate_shuffle_reciprocity_arcs(pairs, targetIntraArcNb)

    def generate_distinct_cluster_spoke_spoke_arcs(self, targetInterArcNb):
        """
        Generate arcs between spokes within distinct clusters.
        """
        if targetInterArcNb == 0:
            return

        clusterNb=len(self.hubsId)
        pairs=[]
        for i in range(clusterNb-1):
            iSpokes = self.clusteredNodesId[i][:-1]
            for j in range(i+1, clusterNb):
                jSpokes = self.clusteredNodesId[j][:-1]
                pairs += self.enumerate_lists_pairs(iSpokes, jSpokes)

        self.generate_shuffle_reciprocity_arcs(pairs, targetInterArcNb)

    def generate_distinct_cluster_hub_spoke_arcs(self, arcBudget):
        clusterNb=len(self.hubsId)
        candidatePairs = []
        for i in range(clusterNb-1):
            candidatePairs += self.enumerate_list_pairs(self.clusteredNodesId[i])
            for j in range(i+1, clusterNb):
                candidatePairs += self.enumerate_lists_pairs(self.clusteredNodesId[i], self.clusteredNodesId[j])
        candidatePairs += self.enumerate_list_pairs(self.clusteredNodesId[clusterNb-1])
        self.generate_shuffle_reciprocity_arcs(candidatePairs, arcBudget)

    def enumerate_list_pairs(self, list):
        """
        Enumerate unconnected candidate node pairs within the same cluster.
        """
        nodePairs = []
        for i in range(len(list)-1):
            for j in range(i+1, len(list)):
                if (list[i],list[j]) in self.pairs or (list[j],list[i]) in self.pairs:
                    continue
                nodePairs.append((list[i],list[j]))
        
        return nodePairs

    def enumerate_lists_pairs(self, list1, list2):
        """
        Enumerate unconnected candidate node pairs between clusters.
        """
        nodePairs = []
        for i in list1:
            for j in list2:
                if (i,j) in self.pairs or (j,i) in self.pairs:
                    continue
                nodePairs.append((i, j))

        return nodePairs

    def generate_shuffle_reciprocity_arcs(self, candidatePairs, maxArcNb):
        """
        Randomly select candidate pairs and generate arcs with reciprocity.
        """
        newArcNb = 0
        k = 0
        random.shuffle(candidatePairs)
        while (newArcNb<maxArcNb and k < len(candidatePairs)):
            (i,j) = candidatePairs[k]
            newArcs = self.reciprocity_arc_generation(True, i, j)
            self.update_arc_pairs(newArcs)
            newArcNb += len(newArcs)
            k += 1

    def generate_cluster_node(self, clusterCenter):
        """
        Sample node coordinates around a cluster center using exponential decay.
        """
        distance = np.random.exponential(scale=1/self.params.decayRate)
        angle = np.random.uniform(0, 2*np.pi)
        x = clusterCenter[0] + distance * np.cos(angle)
        y = clusterCenter[1] + distance * np.sin(angle)
        return (x,y)

    def get_arc_distance(self, iId, jId):
        """
        Compute Euclidean distance between two nodes using bounding box scale if specified.
        """
        width=1
        height=1
        if self.params.bboxWidth is not None and self.params.bboxHeight is not None:
            width=self.params.bboxWidth
            height=self.params.bboxHeight
        xI = self.nodes[iId].x * width
        xJ = self.nodes[jId].x * width
        yI = self.nodes[iId].y * height
        yJ = self.nodes[jId].y * height
        xx = xI - xJ
        yy = yI - yJ
        return math.sqrt(xx**2 + yy**2)

    def get_arc_costs(self, distance):
        """
        Compute fixed and unit costs proportional to arc length.
        """
        fixedCost = round(distance*0.5*60*0.55,5)
        unitCost = round(self.params.ufCostRatio*fixedCost/self.params.capacity, 5)
        return (fixedCost, unitCost)
    
    def get_new_arc(self, idFrom, idTo):
        """
        Create a new directed arc between nodes with distance-based costs.
        """
        distance = self.get_arc_distance(idFrom, idTo)
        fixedCost, unitCost = self.get_arc_costs(distance)
        return Arc(
            src=idFrom,
            dest=idTo,
            unit=unitCost,
            fixed=fixedCost,
            capacity=self.params.capacity,
            distance=distance
        )
    
    def reciprocity_arc_generation(self, useReciprocity, idFrom, idTo):
        """
        Generate arcs between two nodes with or without reciprocity.

        Args:
            useReciprocity (bool): Whether reciprocity is applied, if not both arcs are generated.
            idFrom (int): Source node index.
            idTo (int): Target node index.

        Returns:
            list[Arc]: One or two arcs depending on reciprocity.
        """
        newArcs = []
        if (not useReciprocity) or (random.random() < self.adjustedR):
            newArcs.append(self.get_new_arc(idFrom, idTo))
            newArcs.append(self.get_new_arc(idTo, idFrom))
        else:
            if random.random() < 0.5:
                newArcs.append(self.get_new_arc(idFrom, idTo))
            else:
                newArcs.append(self.get_new_arc(idTo, idFrom))
        return newArcs

    def update_arc_pairs(self, newArcs: list[Arc]):
        """
        Add new arcs and record their directed pairs.
        """
        self.arcs += newArcs
        for arc in newArcs:
            self.pairs.add((arc.src, arc.dest))

    def compute_arc_number(self):
        """
        Compute target number of arcs from either:
            - targetArcNb
            - targetDensity

        Raises:
            ValueError: If neither is defined.
        """
        maxArcNb = self.params.targetNodeNb*(self.params.targetNodeNb-1)
        if (self.params.targetArcNb is not None):
            self.targetArcNb = self.params.targetArcNb
        elif (self.params.targetDensity is not None):
            self.targetArcNb = round(maxArcNb*self.params.targetDensity)
        else:
            raise ValueError("No target density or arc number set.")

    def validity_check(self):
        if self.params.targetNodeNb != len(self.nodes):
            raise Exception(f"Incorrect number of nodes generated: asked={self.params.targetNodeNb} ; generated={len(self.nodes)}")
        
        # Check for duplicates arcs.
        pairCheck = set()
        for arc in self.arcs:
            if arc.src == arc.dest:
                raise Exception("Error: looping arc detected")
            if (arc.src,arc.dest) in pairCheck:
                raise Exception("Error: duplicate arc detected")
            pairCheck.add((arc.src,arc.dest))
    
    def plot_network(self):
        """
        Plots the current nodes of the generator. The plot is saved in the current active directory.
        """
        spokes=[]
        hubs=[]

        for i, cluster in enumerate(self.clusteredNodesId):
            for spokeId in cluster[:-1]:
                spokes.append((self.nodes[spokeId].x, self.nodes[spokeId].y))
            hubs.append((self.nodes[cluster[-1]].x,self.nodes[cluster[-1]].y))

        spokes = np.vstack(spokes)
        hubs = np.vstack(hubs)

        plt.scatter(spokes[:, 0], spokes[:, 1], color='black', marker='x', s=50, edgecolors='white', label="Spokes", zorder=2)
        plt.scatter(hubs[:, 0], hubs[:, 1], color='red', marker='o', s=150, edgecolors='white', label="Hubs", zorder=2)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("generated_network.png", dpi=300, bbox_inches="tight")

    def clear(self):
        self.nodes.clear()
        self.arcs.clear()
        self.pairs.clear()
        self.adjustedR = 0.0
        self.hubsId.clear()
        self.clusteredNodesId.clear()
        self.targetArcNb = 0

    @staticmethod
    def score_pair_degree(nodesDegree, inPairs):
        pairScore = []
        for (i,j) in inPairs:
            pairScore.append(nodesDegree[i] + nodesDegree[j])

        return pairScore