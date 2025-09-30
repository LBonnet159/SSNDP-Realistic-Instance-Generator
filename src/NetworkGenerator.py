import math
import random
import statistics
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
from pathlib import Path
from Structures import Node, Arc, NetworkGeneratorParams, Network

class NetworkGenerator:
    def __init__(self, seed, config: NetworkGeneratorParams):
        self.seed = seed
        self.config = config
        self.nodes: list[Node] = []
        self.arcs: list[Arc] = []
        self.pairs = set()
        self.adjustedR = 0.0
        self.hubsId = []
        self.clusteredNodesId = []
        self.targetArcNb = 0
        self.intraArcsRatio = 1.0

    def emulate_network(self):
        if (self.config.networkEmulationPath is None):
            raise ValueError("Path to network to emulate not specified. Specify in Config.txt file.")

        # Read network to emulate.
        network = Network.from_file(Path(self.config.networkEmulationPath), isArcDistance=True)

        # Retrieve network parameters.
        maxHeight=0
        maxWidth=0
        if network.nodes[0].x is not None:
            for node in network.nodes:
                if node.x > maxWidth:
                    maxWidth = node.x
                if node.y > maxHeight:
                    maxHeight = node.y

        capacities=[]
        priceRatio=[]
        for arc in network.arcs:
            capacities.append(arc.capacity)
            priceRatio.append(arc.unit / arc.fixed)
        avgCapa = statistics.mean(capacities)
        avgPriceRatio = statistics.mean(priceRatio)

        # Build network to emulate in NetworkX.
        G=nx.DiGraph()
        G.add_nodes_from([0, len(network.nodes)-1])
        for arc in network.arcs:
            G.add_edge(arc.src, arc.dest, weight=arc.distance)
        
        # Compute network to emulate CNA metrics.
        density = nx.density(G)
        reciprocity = nx.reciprocity(G)
        avgClusterCoef = nx.clustering(G)
        avgSpl = nx.shortest_path_length(G)

        # Generate networks.
        matchingNetworkFound = False
        timeLimit = 300
        start = time.time()
        end = time.time()
        while not matchingNetworkFound or end-start < timeLimit:
            # Generate, check metrics, loop.
            print("test")

    def generate(self,):
        """
        Network generation main entry-point.
        """

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.compute_arc_number()
        if self.config.targetReciprocity is None:
            self.config.targetReciprocity = self.targetArcNb/(self.config.targetNodeNb*(self.config.targetNodeNb-1))

        if self.config.randomGeneration:
            self.random_generation()
        else:
            self.node_generation()
            self.arc_generation()
            
            if __debug__:
                # Plot network for debugging purposes.
                plotNetwork=False
                if plotNetwork:
                    self.plot_network()

                self.validity_check()

        self.arcs = self.arcs[:self.targetArcNb]

    def save(self, path, folder, id: int):
        if folder is not None:
            path += "/" + folder
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        outPath = path / self.generate_file_name(id)
        f = open(outPath,"w")
        f.write("Nodes,"+str(len(self.nodes))+"\n")
        i = 0
        for node in self.nodes:
            strClusterId = ""
            strX = ""
            strY = ""
            if node.clusterId is not None: strClusterId = str(node.clusterId)
            if node.x is not None: strX = str(node.x)
            if node.y is not None: strY = str(node.y)
            f.write(str(i)+","+strClusterId+","+strX+","+strY+"\n")
            i+=1
        f.write("Arcs,"+str(len(self.arcs))+"\n")
        i = 0
        for arc in self.arcs:
            f.write(str(i)+","+str(arc.src)+","+str(arc.dest)+","+str(arc.unit)+","+str(arc.fixed)+","+str(arc.capacity)+","+str(arc.distance)+"\n")
            i+=1
        f.close()

    def generate_file_name(self, id: int):
        fileName = ""
        if self.config.randomGeneration:
            fileName += "Random"
        else:
            dr=str(self.config.decayRate)
            a=str(int(self.config.hnRatio*100))
            t=str(int(self.config.interArcsRatio*100))
            fileName += "DR"+dr+"_A"+a+"_T"+t

        ufString = str(int(self.config.ufCostRatio*100))
        dString = str(int(self.config.targetDensity*100))
        rString = str(int(self.config.targetReciprocity*100))
        fileName += "_UF"+ufString+"_R"+rString+"_D"+dString+"_N"+str(self.config.targetNodeNb)+"_I"+str(id)

        if self.seed is not None:
            fileName += "_S"+str(self.seed)

        fileName += ".txt"
        return fileName

    def random_generation(self):
        """
        Generates a random network. 
        
        Nodes are generated randomly in the range [0,1].
        Arcs are generated randomly.
        If a reciprocity is given it is used in the generation.
        """

        self.adjustedR = self.config.targetReciprocity/(2-self.config.targetReciprocity)
        random_points = np.random.rand(self.config.targetNodeNb, 2)
        self.nodes.extend(Node(x, y, 0) for x, y in random_points)
        allPairs = list(permutations(range(self.config.targetNodeNb),2))
        random.shuffle(allPairs)
        useReciprocity = self.config.targetReciprocity is not None
        for i, j in allPairs:
            if ((i, j) not in self.pairs) and ((j, i) not in self.pairs):
                newArcs = self.get_new_arcs(useReciprocity, i, j)
                self.update_arc_pairs(newArcs)
                if len(self.arcs) >= self.targetArcNb:
                    break

    def node_generation(self):
        """
        Generate the clustered nodes of the hub-and-spoke network.
        """
         
        # We set the number of clusters.
        clusterNb = math.ceil(self.config.hnRatio * self.config.targetNodeNb)
        if clusterNb == 0:
            clusterNb = 1

        # We generate the center of each cluster, in a [0, 1] x [0, 1] plane.
        clusterCenters = np.random.rand(clusterNb, 2)
        
        # We assign the points to each cluster.
        remainingPointsNb = self.config.targetNodeNb - clusterNb
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
                x = round(x * self.config.rectangleWidth,5)
                y = round(y * self.config.rectangleHeight, 5)
                self.nodes.append(Node(x,y,i))
                
            nodesId.append(len(self.nodes))
            self.hubsId.append(len(self.nodes))
            x = round(clusterCenter[0] * self.config.rectangleWidth, 5)
            y = round(clusterCenter[1] * self.config.rectangleHeight, 5)
            self.nodes.append(Node(x,y,i))
            self.clusteredNodesId.append(nodesId)

    def arc_generation(self):
        """
        Generate the arcs of the hub-and-spoke network.
        Clustered nodes must have been generated by the generator beforehand.
        """
        nodeNb = len(self.nodes)
        if (nodeNb != self.config.targetNodeNb):
            raise Exception(f"Node number produced not correct: asked={self.config.targetNodeNb} ; generated={nodeNb}")

        clusterNb = len(self.clusteredNodesId)
        
        # We compute if the arc budget is enough to produce a weakly connected graph.
        minBackboneArcNb = 2*clusterNb
        minSpokeHubArcNb = math.ceil((nodeNb-clusterNb)*(self.config.targetReciprocity+1))
        if (minBackboneArcNb+minSpokeHubArcNb > self.targetArcNb):
            raise Exception(f"Arc budget too low. Increase node number or density, or lower the reciprocity or the hub and spokes ratio. " \
            "Budget={self.targetArcNb} ; Minimum arc needed={minBackboneArcNb+minSpokeHubArcNb}")

        # Step 1: We generate reciprocal arcs between hubs.
        self.generate_hub_hub_arcs()
        arcBudget = self.targetArcNb - len(self.arcs)

        # We adjust the target reciprocity.
        self.adjustedR = (self.targetArcNb*self.config.targetReciprocity-len(self.arcs))/arcBudget
        self.adjustedR = self.adjustedR/(2-self.adjustedR)
        self.adjustedR = max(0,self.adjustedR)

        # Step 2: We generate arcs between spokes and hubs.
        self.generate_spoke_hub_arcs()
        arcBudget = self.targetArcNb - len(self.arcs)

        # Step 3+4: We generate intra and inter cluster spoke to spoke arcs.
        intraArcNb, interArcNb = self.compute_special_arc_nb()
        if intraArcNb > 0:
            self.generate_intra_arcs(intraArcNb)
        if len(self.hubsId) > 1 and interArcNb > 0:
            self.generate_inter_arcs(interArcNb)
        
        arcBudget = self.targetArcNb - len(self.arcs)
        self.generate_random_arcs(arcBudget)

    def generate_hub_hub_arcs(self):
        """
        Backbone generation, link hub spokes to other hub spokes.
        """
        hubNb = len(self.hubsId)
        maxBackboneArcNb = hubNb*(hubNb-1)
        spokeNb = len(self.nodes)-hubNb
        maxSpokeHubArcNb = math.ceil(spokeNb*(self.config.targetReciprocity+1))
        neighborNb = hubNb-1
        if maxBackboneArcNb+maxSpokeHubArcNb > self.targetArcNb:
            neighborNb -= 1
            while neighborNb >= 2:
                newBackboneArcNb = hubNb*neighborNb
                if newBackboneArcNb + maxSpokeHubArcNb <= self.targetArcNb:
                    break
                neighborNb -= 1

        if neighborNb >= len(self.hubsId):
            raise Exception(f"Hub asked to be linked to more hubs ({neighborNb}) than possible ({len(self.hubsId)}).")
        
        for i in range(len(self.hubsId)):
            for j in range(1, neighborNb+1):
                u, v = self.hubsId[i], self.hubsId[(i+j) % len(self.hubsId)]
                if (u,v) in self.pairs:
                    continue
                newArcs = self.get_new_arcs(False, u, v)
                self.update_arc_pairs(newArcs)

    def generate_spoke_hub_arcs(self):
        """
        Generates arcs between spoke nodes and their hub node.
        """
        for i in range(len(self.hubsId)):
            spokeHubPairs = [(spoke, self.hubsId[i]) for spoke in self.clusteredNodesId[i][:-1]] # List spoke hub pairs.
            self.generate_shuffle_reciprocity_arcs(spokeHubPairs, float('inf'))

    def compute_special_arc_nb(self):
        """
        Computes the number of inter and intra arcs to generate.
        """
        arcBudget = self.targetArcNb - len(self.arcs)
        clusterNb = len(self.hubsId)

        # Compute max possible number of intra and inter arcs to generate.
        maxIntraArcNb = 0
        maxInterArcNb = 0
        for i in range(clusterNb-1):
            iSpokeNb = len(self.clusteredNodesId[i])-1
            maxIntraArcNb += iSpokeNb * (iSpokeNb-1)
            for j in range(i+1, clusterNb):
                jSpokeNb = len(self.clusteredNodesId[j])-1
                maxInterArcNb += iSpokeNb * jSpokeNb * 2
        lastClusterSpokeNb = len(self.clusteredNodesId[clusterNb-1])-1
        maxIntraArcNb += lastClusterSpokeNb * (lastClusterSpokeNb-1)

        # Compute intra and inter arcs to generate according to user ratio.
        targetIntraArcsNb = round(maxIntraArcNb*self.intraArcsRatio)
        targetInterArcsNb = round(maxInterArcNb*self.config.interArcsRatio)

        # Handle case not enough arc budget.
        if targetIntraArcsNb > arcBudget and targetInterArcsNb <= 0:
            targetIntraArcsNb = arcBudget
        elif targetInterArcsNb > arcBudget and targetIntraArcsNb <= 0:
            targetInterArcsNb = arcBudget
        elif targetIntraArcsNb+targetInterArcsNb > arcBudget:
            # Normalize and scale down number of inter and intra arcs to generate.
            targetIntraArcsNb = math.floor((targetIntraArcsNb*arcBudget)/(targetIntraArcsNb+targetInterArcsNb))
            #targetInterArcsNb = arcBudget - targetIntraArcsNb

        targetInterArcsNb = arcBudget - targetIntraArcsNb

        return targetIntraArcsNb, targetInterArcsNb
    
    def generate_intra_arcs(self, targetIntraArcNb):
        """
        Generate intra arcs (i.e. arcs between spokes of the same cluster).
        """
        if targetIntraArcNb == 0:
            return

        hubNb=len(self.hubsId)
        arcNbPerCluster = math.ceil(targetIntraArcNb / hubNb)

        for i in range(len(self.clusteredNodesId)):
            spokes = self.clusteredNodesId[i][:-1]
            spokesNb = len(spokes)
            clusterMaxArcNb = spokesNb*(spokesNb-1)*self.intraArcsRatio

            # Enumerate pairs.
            pairs = []
            for i in range(len(spokes)):
                for j in range(i + 1, len(spokes)): 
                    pairs.append((spokes[i], spokes[j]))
            random.shuffle(pairs)
            
            """
            random.shuffle(spokes)
            nodePairs = []
            for k in range(len(spokes)-1):
                for l in range(k+1, len(spokes)):
                    if (spokes[k],spokes[l]) in self.pairs or (spokes[l],spokes[k]) in self.pairs:
                        continue

                    nodePairs.append((spokes[k],spokes[l]))
                if (len(nodePairs) > clusterMaxArcNb):
                    break"""

            self.generate_shuffle_reciprocity_arcs(pairs, arcNbPerCluster)

    def generate_inter_arcs(self, targetInterArcNb):
        """
        Generate inter arcs (i.e. arcs between spokes of different cluster).
        Arcs are generated by linking spokes of high degree.
        """
        if targetInterArcNb == 0:
            return

        clusterNb=len(self.hubsId)
        clusterPairNb = clusterNb * (clusterNb-1) / 2
        arcNbPerClusterPair = math.ceil(targetInterArcNb / clusterPairNb)

        # Compute nodes degree.
        nodesDegree = [0 for _ in range(self.config.targetNodeNb)]
        for arc in self.arcs:
            nodesDegree[arc.src] += 1
            nodesDegree[arc.dest] += 1

        # Select spoke nodes to consider for each cluster by degree.
        sortedSpokes = []
        selectedSpokes = []
        for cluster in self.clusteredNodesId:
            spokes = cluster[:-1]
            spokes = sorted(spokes, key=lambda n: nodesDegree[n], reverse=True)
            sortedSpokes.append(spokes)
            keepNb = max(1, math.ceil(len(spokes) * self.config.connectedSpokesRatio))
            selectedSpokes.append(spokes[:keepNb])
        
        """
        # Score candidate pairs by degree. Choose randomly which pairs to link among highest degree pairs.
        for i in range(clusterNb-1):
            iSpokes = selectedSpokes[i]
            for j in range(i+1, clusterNb):
                jSpokes = selectedSpokes[j]

                # Adapt number of selected spokes if not sufficient.
                minPairNb = arcNbPerClusterPair / (1+self.adjustedR)
                minINb = math.ceil(minPairNb/max(1,len(iSpokes)))
                minJNb = math.ceil(minPairNb/max(1,len(jSpokes)))
                if minINb > len(iSpokes):
                    iSpokes = sortedSpokes[i][:min(minINb,len(sortedSpokes[i]))]
                if minJNb > len(jSpokes):
                    jSpokes = sortedSpokes[j][:min(minJNb,len(sortedSpokes[j]))]

                candidatePairs = self.enumerate_lists_pairs(iSpokes, jSpokes)
                random.shuffle(candidatePairs)
                self.generate_shuffle_reciprocity_arcs(candidatePairs, arcNbPerClusterPair)"""
        
        for i in range(clusterNb-1):
            iSpokes = self.clusteredNodesId[i][:-1]
            for j in range(i+1, clusterNb):
                jSpokes = self.clusteredNodesId[j][:-1]
                interClusterPairs = self.enumerate_lists_pairs(iSpokes, jSpokes)
                pairsScores = self.score_pair_degree(nodesDegree, interClusterPairs)
                sorted_pairs = [pair for _, pair in sorted(zip(pairsScores, interClusterPairs), reverse=True)]
                topDegreePairs = sorted_pairs[:arcNbPerClusterPair]
                self.generate_shuffle_reciprocity_arcs(topDegreePairs, arcNbPerClusterPair)

    def generate_random_arcs(self, arcBudget):
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
        Enumerate candidate node pairs not yet linked
        of the same cluster.
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
        Enumerate candidate node pairs not yet linked
        between two clusters.
        """
        nodePairs = []
        for i in list1:
            for j in list2:
                if (i,j) in self.pairs or (j,i) in self.pairs:
                    continue
                nodePairs.append((i, j))

        return nodePairs

    def generate_shuffle_reciprocity_arcs(self, candidatePairs, maxArcNb):
        newArcNb = 0
        k = 0
        random.shuffle(candidatePairs)
        while (newArcNb<maxArcNb and k < len(candidatePairs)):
            (i,j) = candidatePairs[k]
        
            newArcs = self.get_new_arcs(True, i, j)
            self.update_arc_pairs(newArcs)
            newArcNb += len(newArcs)
            k += 1

    def generate_cluster_node(self, clusterCenter):
        distance = np.random.exponential(scale=1/self.config.decayRate)
        angle = np.random.uniform(0, 2*np.pi)
        x = clusterCenter[0] + distance * np.cos(angle)
        y = clusterCenter[1] + distance * np.sin(angle)
        return (x,y)

    def get_arc_distance(self, iId, jId):
        xI = self.nodes[iId].x * self.config.rectangleWidth
        xJ = self.nodes[jId].x * self.config.rectangleWidth
        yI = self.nodes[iId].y * self.config.rectangleHeight
        yJ = self.nodes[jId].y * self.config.rectangleHeight
        xx = xI - xJ
        yy = yI - yJ
        return math.sqrt(xx**2 + yy**2)

    def get_arc_costs(self, distance):
        fixedCost = round(distance*0.5*60*0.55,5)
        unitCost = round(self.config.ufCostRatio*fixedCost/self.config.capacity, 5)
        return (fixedCost, unitCost)
    
    def get_new_arc(self, idFrom, idTo):
        distance = self.get_arc_distance(idFrom, idTo)
        fixedCost, unitCost = self.get_arc_costs(distance)
        return Arc(
            src=idFrom,
            dest=idTo,
            unit=unitCost,
            fixed=fixedCost,
            capacity=self.config.capacity,
            distance=distance
        )
    
    def get_new_arcs(self, useReciprocity, idFrom, idTo):
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
        self.arcs += newArcs
        for arc in newArcs:
            self.pairs.add((arc.src, arc.dest))

    def compute_arc_number(self):
        # We compute the number of arcs to produce.
        maxArcNb = self.config.targetNodeNb*(self.config.targetNodeNb-1)
        if (self.config.targetArcNb is not None):
            self.targetArcNb = self.config.targetArcNb
        elif (self.config.targetDensity is not None):
            self.targetArcNb = round(maxArcNb*self.config.targetDensity)
        else:
            raise ValueError("No target density or arc number set.")

    def validity_check(self):
        if self.config.targetNodeNb != len(self.nodes):
            raise Exception(f"Incorrect number of nodes generated: asked={self.config.targetNodeNb} ; generated={len(self.nodes)}")
        
        # Check for duplicates arcs.
        pairCheck = set()
        for arc in self.arcs:
            if (arc.src,arc.dest) in pairCheck:
                raise Exception("Arc duplicate error.")
            pairCheck.add((arc.src,arc.dest))

    @staticmethod
    def score_pair_degree(nodesDegree, inPairs):
        pairScore = []
        for (i,j) in inPairs:
            pairScore.append(nodesDegree[i] + nodesDegree[j])

        return pairScore
    
    def plot_network(self):

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
        plt.savefig("network.png", dpi=300, bbox_inches="tight")