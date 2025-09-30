import os
import ast
from Structures import ConfigParams, NetworkGeneratorParams, InstanceGeneratorParams

class Config:
    def __init__(self, params: dict):
        self.configParams: ConfigParams = Config.get_config_parameters(params)
        self.networkGeneratorParams: NetworkGeneratorParams = Config.get_network_params(params)
        self.instanceGeneratorParams: InstanceGeneratorParams = Config.get_instance_params(params)

    @staticmethod
    def from_file(path: str = "src/Config.txt"):
        """
        To read the .txt config file.
        """
        params = Config.parse_config_file(path)
        return Config(params)

    @staticmethod
    def parse_config_file(path: str):
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
                            params[key] = ast.literal_eval(value)  # safely parse list
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
        return ConfigParams(
            defaultInstancePath=params.get('defaultInstancePath'),
            defaultNetworkPath=params.get('defaultNetworkPath'),
            folder=params.get('folder'),
            networkNb=params.get('networkNb'),
            instanceNb=params.get('instanceNb'),
            preProcessingSSNDP=params.get('preProcessingSSNDP'),
            doStatic=params.get('doStatic'),
            networkSeed=params.get('networkSeed'),
            demandSeed=params.get('demandSeed')
        )
    
    @staticmethod
    def get_network_params(params):
        return NetworkGeneratorParams(
            networkEmulationPath=params.get('networkEmulationPath'),
            randomGeneration=params.get('randomGeneration'),
            capacity=params.get('capacity'),
            rectangleWidth=params.get('rectangleWidth'),
            rectangleHeight=params.get('rectangleHeight'),
            targetNodeNb=params.get('targetNodeNb'),
            targetArcNb=params.get('targetArcNb'),
            targetDensity=params.get('targetDensity'),
            targetReciprocity=params.get('targetReciprocity'),
            decayRate=params.get('decayRate'),
            hnRatio=params.get('hnRatio'),
            interArcsRatio=params.get('interArcsRatio'),
            connectedSpokesRatio=params.get('connectedSpokesRatio'),
            ufCostRatio=params.get('ufCostRatio')
        )
    
    @staticmethod
    def get_instance_params(params):
        return InstanceGeneratorParams(
            doStatic=params.get('doStatic'),
            commodityNb=params.get('commodityNb'),
            quantityToCapacityMean=params.get('quantityToCapacityMean'),
            quantityToCapacityDev=params.get('quantityToCapacityDev'),
            days=params.get('days'),
            hours=params.get('hours'),
            flexibilityParameterMean=params.get('flexibilityParameterMean'),
            flexibilityParameterDev=params.get('flexibilityParameterDev'),
            criticalTime=params.get('criticalTime')
        )