from Config import Config
from NetworkGenerator import NetworkGenerator
from InstanceGenerator import InstanceGenerator
from Structures import Network, Instance
from pathlib import Path

def main():
    config = Config.from_file()

    # We generate networks.
    if config.configParams.networkNb > 0:
        currentSeed = config.configParams.networkSeed
        for i in range(config.configParams.networkNb):
            networkGenerator = NetworkGenerator(currentSeed, config.networkGeneratorParams)
            networkGenerator.generate()
            networkGenerator.save(path=config.configParams.defaultNetworkPath ,folder=config.configParams.folder, id=i)
            if currentSeed is not None:
                currentSeed+=1

    # We generate (S)SNDP instances.
    if config.configParams.instanceNb > 0:
        path = config.configParams.defaultNetworkPath
        if len(config.configParams.folder) > 0:
            path += "/" + config.configParams.folder
        path = Path(path)
        for networkFile in path.glob("*.txt"):
            network = Network.from_file(networkFile, isArcDistance=True)
            currentSeed = config.configParams.networkSeed
            for i in range(config.configParams.instanceNb):
                instanceGenerator = InstanceGenerator(currentSeed, config.instanceGeneratorParams, network)
                instanceGenerator.generate()
                instanceGenerator.save(path=config.configParams.defaultInstancePath ,folder=config.configParams.folder, 
                                  basename=networkFile.name, id=i)
                if currentSeed is not None:
                    currentSeed+=1

    # We generate preprocessing files for target instances.
    if config.configParams.preProcessingSSNDP:
        path = config.configParams.defaultNetworkPath
        if len(config.configParams.folder) > 0:
            path += "/" + config.configParams.folder
        path = Path(path)
        for instanceFile in path.glob("*.txt"):
            instance = Instance.from_file(instanceFile)

if __name__ == "__main__":
    main()