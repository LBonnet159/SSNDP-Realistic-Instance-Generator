from Config import Config
from NetworkGenerator import NetworkGenerator
from InstanceGenerator import InstanceGenerator
from Structures import Network
from pathlib import Path

def main():
    config = Config.from_file()

    # Resolve save paths.
    networkPath = Path(config.configParams.defaultNetworkPath)
    instancePath = Path(config.configParams.defaultInstancePath)
    if config.configParams.folder:
        networkPath = networkPath / config.configParams.folder
        instancePath = instancePath / config.configParams.folder

    print("Start of generator.")
    print(f"General configuration parameters:\n{config.configParams}")

    # We generate networks. They are saved at defaultNetworkPath\folder.
    if config.configParams.networkNb > 0:
        print("\nStart of network generation.")

        # Set generator.
        networkSavedNb = 0
        currentSeed = config.configParams.networkSeed
        networkParams = config.get_network_params_combinations()
        for params in networkParams:
            print(f"Network generation parameters:\n{params}")
            networkGenerator = NetworkGenerator(currentSeed, params)
            if (config.networkGeneratorParams.networkEmulationPath is not None):
                # Emulate specified network.
                networks = networkGenerator.emulate_network(config.configParams.networkNb)
                for network in networks:
                    network.save(path=networkPath)
                    networkSavedNb+=1
            else:
                # Generate networks.
                for i in range(config.configParams.networkNb):
                    network = networkGenerator.generate()
                    network.save(path=networkPath)
                    networkSavedNb+=1

        print("\nNetwork generation over.")
        print(f"Number of network(s) saved: {networkSavedNb}")
        print(f"Network(s) saved at path: {networkPath.resolve()}")

    # We generate (S)SNDP instances for each network of the specified defaultNetworkPath\folder.
    if config.configParams.instanceNb > 0:
        print("\nStart of instance generation.")

        # Resolve network input path.
        networkPath = Path(config.configParams.defaultNetworkPath)
        if config.configParams.folder:
            networkPath = networkPath / config.configParams.folder

        # Generate instance(s) for all network(s) at specified path.
        instanceSavedNb = 0        
        instanceParams = config.get_instance_params_combinations()
        for params in instanceParams:
            print(f"Instance generation parameters:\n{params}")
            for networkFile in networkPath.glob("*.txt"):
                network = Network.from_file(networkFile, isArcDistance=True)
                currentSeed = config.configParams.networkSeed
                for i in range(config.configParams.instanceNb):
                    instanceGenerator = InstanceGenerator(currentSeed, params, network)
                    instanceGenerator.generate()
                    instanceGenerator.save(path=config.configParams.defaultInstancePath ,folder=config.configParams.folder, 
                                    basename=networkFile.name, id=i)
                    instanceSavedNb += 1
                    if currentSeed is not None:
                        currentSeed+=1
        
        print("\nInstance generation over.")
        print(f"Number of instance(s) saved: {instanceSavedNb}")
        print(f"Instance(s) saved at path: {instancePath.resolve()}")

if __name__ == "__main__":
    main()