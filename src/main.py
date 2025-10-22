from Config import Config
from NetworkGenerator import NetworkGenerator
from InstanceGenerator import InstanceGenerator
from Structures import Network
from pathlib import Path

def main():
    """
    Main entry point for the (S)SNDP realistic instance generator.

    This script performs the generation process in three main steps:
    1. Load configuration parameters from the configuration file.
    2. Generate synthetic or emulated networks according to specified CNA parameters.
    3. Generate SNDP or SSNDP instances based on those networks. Optionally with pre-processings.

    Usage
    -----
    Simply run:
        python main.py

    The configuration file Config.txt defines all parameter combinations and paths, including:
    - Number of networks and instances to generate
    - Network structure options (hub ratio, reciprocity...)
    - Instance options (number of commodities, horizon....)
    - Folder and base path locations for saving output and input path for emulation

    Outputs
    -------
    - Network files (.txt) stored in `defaultNetworkPath[/folder]`
    - Instance files (.txt) stored in `defaultInstancePath[/folder]`

    Notes
    -----
    - Networks can either be generated procedurally (in a random or hub-and-spoke way) or emulated from a saved network file.
    - Instance generation supports both static (SNDP) and time-dependent (SSNDP) variants.
    - Each combination of parameters produces multiple runs if requested.
    - Parameters can be either given a single value, or a list of values for multiple parameter combination.
    """
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