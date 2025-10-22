import itertools
import ast
from copy import deepcopy
from Structures import ConfigParams, NetworkGeneratorParams, InstanceGeneratorParams

class Config:
    def __init__(self, params: dict):
        self.configParams: ConfigParams = Config.get_config_parameters(params)
        self.networkGeneratorParams: NetworkGeneratorParams = Config.get_network_params(params)
        self.instanceGeneratorParams: InstanceGeneratorParams = Config.get_instance_params(params)

    @staticmethod
    def from_file(path: str = "src/Config.txt"):
        """
        Config file based constructor.
        """
        params = Config.parse_config_file(path)
        return Config(params)

    @staticmethod
    def parse_config_file(path: str):
        """
        Config file parser.
        """
        params = {}
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        try:
                            params[key] = ast.literal_eval(value)  # Parse list.
                        except Exception:
                            params[key] = value
                    elif len(value) == 0 or value == 'None':
                        params[key] = None
                    elif value.lower() == "true":
                        params[key] = True
                    elif value.lower() == "false":
                        params[key] = False
                    elif value.isdigit():
                        params[key] = int(value)
                    else:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value

        return params

    @staticmethod
    def get_config_parameters(params):
        """
        Returns general configuration parameters.
        """
        return ConfigParams(
            defaultInstancePath=params.get('defaultInstancePath'),
            defaultNetworkPath=params.get('defaultNetworkPath'),
            folder=params.get('folder'),
            networkNb=params.get('networkNb'),
            instanceNb=params.get('instanceNb'),
            networkSeed=params.get('networkSeed'),
            demandSeed=params.get('demandSeed')
        )
    
    @staticmethod
    def get_network_params(params):
        """
        Returns network generation related parameters.
        """
        return NetworkGeneratorParams(
            networkEmulationPath=params.get('networkEmulationPath'),
            networkEmulationTimeLimit=params.get('networkEmulationTimeLimit'),
            randomGeneration=params.get('randomGeneration'),
            capacity=params.get('capacity'),
            bboxWidth=params.get('bboxWidth'),
            bboxHeight=params.get('bboxHeight'),
            targetNodeNb=params.get('targetNodeNb'),
            targetArcNb=params.get('targetArcNb'),
            targetDensity=params.get('targetDensity'),
            targetReciprocity=params.get('targetReciprocity'),
            decayRate=params.get('decayRate'),
            hnRatio=params.get('hnRatio'),
            ufCostRatio=params.get('ufCostRatio')
        )
    
    @staticmethod
    def get_instance_params(params):
        """
        Returns instance generation related parameters.
        """
        return InstanceGeneratorParams(
            doStatic=params.get('doStatic'),
            commodityNb=params.get('commodityNb'),
            quantityToCapaMean=params.get('quantityToCapaMean'),
            quantityToCapaDev=params.get('quantityToCapaDev'),
            sameRegionRatio=params.get('sameRegionRatio'),
            disparityRatio=params.get('disparityRatio'),
            horizon=params.get('horizon'),
            flexibilityMean=params.get('flexibilityMean'),
            flexibilityDev=params.get('flexibilityDev'),
            criticalTime=params.get('criticalTime'),
            distributionPattern=params.get('distributionPattern'),
            preProcessingSSNDP=params.get('preProcessingSSNDP')
        )
    
    def get_network_params_combinations(self):
        """
        Returns all combinations of network parameters if some of them are arrays.
        """
        sweepParams = {name: value for name, value in vars(self.networkGeneratorParams).items()
                       if isinstance(value, list)}
        if not sweepParams:
            return [self.networkGeneratorParams]
        else:
            # Create all combinations
            keys = list(sweepParams.keys())
            values_product = itertools.product(*sweepParams.values())
            runs = []
            for combination in values_product:
                newParams = deepcopy(self.networkGeneratorParams)
                for key, val in zip(keys, combination):
                    setattr(newParams, key, val)
                runs.append(newParams)
            return runs
        
    def get_instance_params_combinations(self):
        """
        Returns all combinations of instqnce parameters if some of them are arrays.
        """
        sweepParams = {name: value for name, value in vars(self.instanceGeneratorParams).items()
                    if isinstance(value, list) and name!="distributionPattern"}
        if not sweepParams:
            return [self.instanceGeneratorParams]
        else:
            # Create all combinations
            keys = list(sweepParams.keys())
            values_product = itertools.product(*sweepParams.values())
            runs = []
            for combination in values_product:
                newParams = deepcopy(self.instanceGeneratorParams)
                for key, val in zip(keys, combination):
                    setattr(newParams, key, val)
                runs.append(newParams)
            return runs