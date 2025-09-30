import random
import math
import numpy as np
import scipy.stats as stats
from scipy.sparse.csgraph import dijkstra
from pathlib import Path

from Structures import Arc, Network, Instance, Commodity, InstanceGeneratorParams

class InstanceGenerator:
    def __init__(self, seed, config: InstanceGeneratorParams, network: Network):
        self.seed = seed
        self.config = config
        self.network = network
        self.timedArcs: list[Arc] = []
        self.commodities: list[Commodity] = []
        self.commoditySet = set()
        self.horizon = self.config.days * self.config.hours

    def generate(self):
        """
        Generate sndp or ssndp instance based on input network. 
        """
        nodeNb = len(self.network.nodes)
        arcNb = len(self.network.arcs)

        # Find longest arc.
        longestArcDist = 0.0
        for arc in self.network.arcs:
            if arc.distance > longestArcDist:
                longestArcDist = arc.distance

        # Set longest (in distance) arc to take 0.2 of the horizon (in time). Apply to all arcs. 
        convRate = (self.horizon*0.2)/longestArcDist
        for arc in self.network.arcs:
            arc.time = math.ceil(arc.distance*convRate)

        # Generate quantity of all commodities.
        quantities = self.generate_commodities_quantities()

        # Compute travel times between all pairs.
        travelTimes = self.compute_travel_times()
        validPairs = list(map(tuple, np.argwhere((travelTimes != np.inf) & (travelTimes != 0))))
        meanTravelTime = 0.0
        for pair in validPairs:
            meanTravelTime += travelTimes[pair[0],pair[1]]
        meanTravelTime /= len(validPairs)

        # Generate commodities.
        if self.config.doStatic:
            pairs = random.sample(validPairs, self.config.commodityNb)
            self.commodities.extend(Commodity(src, dest, quantity) for (src, dest), quantity in zip(pairs, quantities))
        else:
            flexibleTimes = self.generate_flexible_times(meanTravelTime)
            for quantity, flexibleTime in zip(quantities, flexibleTimes):
                src, dest, availableTime, dueTime = self.unique_timed_commodity(validPairs, travelTimes, flexibleTime)
                self.commodities.append(Commodity(src, dest, quantity, availableTime, dueTime))
                self.commoditySet.add((src, dest, availableTime, dueTime))

    def save(self, path, folder, basename: str, id: int):
        if folder is not None:
            path += "/" + folder
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        outPath = path / self.generate_file_name(basename, id)
        f = open(outPath,"w")
        f.write("Nodes,"+str(len(self.network.nodes))+"\n")
        i = 0
        for node in self.network.nodes:
            strClusterId = ""
            strX = ""
            strY = ""
            if node.clusterId is not None: strClusterId = str(node.clusterId)
            if node.x is not None: strX = str(node.x)
            if node.y is not None: strY = str(node.y)
            f.write(str(i)+","+strClusterId+","+strX+","+strY+"\n")
            i+=1
        f.write("Arcs,"+str(len(self.network.arcs))+"\n")
        i = 0
        for arc in self.network.arcs:
            f.write(str(i)+","+str(arc.src)+","+str(arc.dest)+","+str(arc.unit)+","+str(arc.fixed)+","+str(arc.capacity))
            if not self.config.doStatic:
                f.write(","+str(arc.time))
            f.write("\n")
            i+=1
        f.write("Commodities,"+str(len(self.commodities))+"\n")
        i = 0
        for commodity in self.commodities:
            f.write(str(i)+","+str(commodity.origin)+","+str(commodity.destination)+","+str(commodity.quantity))
            if not self.config.doStatic:
                f.write(","+str(commodity.availableTime)+","+str(commodity.dueTime))
            f.write("\n")
            i+=1
        if not self.config.doStatic:
            f.write("horizon="+str(self.horizon))
        f.close()

    def generate_file_name(self, basename: str, id: int):
        fileName = ""
        if not self.config.doStatic:
            fileName+="S"

        mcq=str(int(self.config.quantityToCapacityMean*100))
        dqv=str(int(self.config.quantityToCapacityDev*100))
        fileName+="SNDP"+"_MCQ"+mcq+"_DQV"+dqv

        if not self.config.doStatic:
            fm=str(int(self.config.flexibilityParameterMean*100))
            fd=str(int(self.config.flexibilityParameterDev*100))
            ct=str(int(self.config.criticalTime))
            fileName += "_D"+str(self.config.days)+"_H"+str(self.config.hours)+"_FM"+fm+"_FR"+fd+"_CT"+ct

        fileName += "_C"+str(self.config.commodityNb)+"_I"+str(id)+"_"+basename
        return fileName


    def unique_timed_commodity(self, validPairs, travelTimes, flexibleTime):
        """
        Returns a unique timed commodity.
        """
        while True:
            src, dest, availableTime, dueTime = self.generate_timed_commodity(validPairs, travelTimes, flexibleTime)
            if (src, dest, availableTime, dueTime) not in self.commoditySet:
                return src, dest, availableTime, dueTime

    def generate_timed_commodity(self, validPairs, travelTimes, flexibleTime):
        """
        Generates the source, destination, available and due time of a commodity.
        """
        src, dest = random.choice(validPairs)
        upperTimeBound = np.ceil(self.horizon - 1 - travelTimes[src][dest] - flexibleTime).astype(int)
        availableTime = random.randint(0, upperTimeBound)
        dueTime = np.ceil(availableTime + travelTimes[src][dest] + flexibleTime).astype(int)
        if (dueTime >= self.horizon):
            raise Exception(f"Due time ({dueTime}) larger than horizon ({self.horizon}).")

        if self.config.criticalTime > 1:
            # We round available and due times to their closest critical time.
            availableTime = math.floor(availableTime / self.config.criticalTime) * self.config.criticalTime
            dueTime = math.ceil(dueTime / self.config.criticalTime) * self.config.criticalTime

        return (src, dest, availableTime, dueTime)

    def generate_flexible_times(self, meanTravelTime):
        meanFlex=meanTravelTime*self.config.flexibilityParameterMean
        stdDevFlex=meanFlex*self.config.flexibilityParameterDev
        flexibleTimes = np.random.normal(loc=meanFlex, scale=stdDevFlex, size=self.config.commodityNb)
        flexibleTimes = np.ceil(np.maximum(flexibleTimes, 0)).astype(int) # Set value below 0 to 0.
        return flexibleTimes

    def generate_commodities_quantities(self):
        """
        Generate commodities quantity as a truncated normal distribution.
        """
        minCapacity = min([arc.capacity for arc in self.network.arcs])
        meanQuantity = minCapacity * self.config.quantityToCapacityMean
        stdDevQuantity = meanQuantity * self.config.quantityToCapacityDev
        minQuantity = 0.01*minCapacity
        truncArg1 = (minQuantity - meanQuantity) / stdDevQuantity
        truncArg2 = (minCapacity - meanQuantity) / stdDevQuantity
        quantity = stats.truncnorm.rvs(truncArg1, truncArg2, loc=meanQuantity, scale=stdDevQuantity, size=self.config.commodityNb)
        return np.round(quantity, decimals=5)

    def compute_travel_times(self):
        """
        Compute travel time between all pairs.
        """
        allPairTime = InstanceGenerator.compute_all_pair_time(len(self.network.nodes), self.network.arcs)


        # Check horizon is valid.
        maxFiniteDistances=[]
        for item in allPairTime.tolist():
            maxFiniteDistances.append(max(d for d in item if d != np.inf))
        maxPath=max(d for d in maxFiniteDistances if d != np.inf)
        if (maxPath>=self.horizon):
            raise Exception(f"Horizon not large enough for the size of the time-expanded network." \
            "Longest path = {maxPath} ; Horizon = {horizon}. Reduce network or enlarge horizon.")
        
        return allPairTime
    
    @staticmethod
    def compute_all_pair_time(nodeNb, arcs: list[Arc]):
        arcNb = len(arcs)

        adjMatrix = np.full((nodeNb, nodeNb), np.inf)
        for i in range(arcNb):
            adjMatrix[arcs[i].src][arcs[i].dest] = arcs[i].time

        return dijkstra(csgraph=adjMatrix, directed=True)

    @staticmethod
    def generate_preprocessings(instance: Instance):
        """
        Pre-processes a SSNDP instance.
        """
        nodeNb = len(instance.nodes)
        arcNb = len(instance.arcs)
        commodityNb = len(instance.commodities)

        # We compute the transit time of all node pairs.
        allPairTime = InstanceGenerator.compute_all_pair_time(nodeNb, instance.arcs)

        # We compute the time windows of each commodity to each node in the static network.
        timeWindowsNode = [[None for _ in range(nodeNb)] for _ in range(commodityNb)]
        for i in range(commodityNb):
            currentK = instance.commodities[i]
            for j in range(nodeNb):
                lb = currentK.availableTime + allPairTime[currentK.origin, j]
                ub = currentK.dueTime - allPairTime[j, currentK.destination]

                # The time window is empty, the commodity can't reach the node.
                if lb > ub:
                    continue

                timeWindowsNode[i][j] = (lb,ub)

        # We compute the time windows of each commodity to each arc in the static network.
        timeWindowsArc = [[None for _ in range(arcNb)] for _ in range(commodityNb)]
        for i in range(commodityNb):
            currentK = instance.commodities[i]
            for j in range(arcNb):
                currentArc = instance.arcs[j]

                lb = timeWindowsNode[i][currentArc.src][0]
                ub = timeWindowsNode[i][currentArc.dest][1] - currentArc.time

                # Empty time window.
                if (timeWindowsNode[i][currentArc.src] is None) or (timeWindowsNode[i][currentArc.dest] is None):
                    continue

                # Commodity arriving after the latest possible time.
                if timeWindowsNode[i][currentArc.src][0] + currentArc.time > timeWindowsNode[i][currentArc.dest][1]:
                    continue

                # Commodity doesn't need to leave its destination or enter its origin.
                if currentArc.src == currentK.destination or currentArc.dest == currentArc.src:
                    continue

                timeWindowsArc[i][j] = (lb,ub)