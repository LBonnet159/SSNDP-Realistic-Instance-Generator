# An Open-Source Generator for Realistic Instances of the Scheduled Service Network Design Problem

This archive is distributed under the [MIT license](LICENSE).

The software and data in this repository are used in the research reported on in the paper "An Open-Source Generator for Realistic Instances of the Scheduled Service Network Design Problem" by L. Bonnet, S. Belieres, M. Hewitt, and S. U. Ngueveu, prior to final acceptance at the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc).

## Cite
To cite the contents of this repository, please cite both the paper and the snapshot of the repository, using their respective DOIs:

https://doi.org/10.1287/ijoc.2025.1704

https://doi.org/10.1287/ijoc.2025.1704.cd

Below is the Bibtex for citing the snapshot of the repository:
```
@misc{BonnetBelieresHewittNgueveu2026,
  author =        {Louis Bonnet and Simon Belieres and Mike Hewitt and Sandra Ulrich Ngueveu},
  publisher =     {INFORMS Journal on Computing},
  title =         {An Open-Source Generator for Realistic Instances of the Scheduled Service Network Design Problem},
  year =          {2026},
  doi =           {10.1287/ijoc.2025.1704.cd},
  url =           {https://github.com/INFORMSJoC/2025.1704},
  note =          {Available for download at https://github.com/INFORMSJoC/2025.1704},
}
```

## Description
The Service Network Design Problem (SNDP), and its timed variant the Scheduled SNDP (SSNDP), are challenging optimization problems arising in freight transportation systems. This software generates instances with hub-and-spoke networks for these problems with some of its parameters based on metrics from the literature on Complex Networks and Complex Networks Analysis.

Each instance, either of the SNDP or the SSNDP, consists of:
+ A __directed network__ with a node set and an arc set
+ A __set of commodities__

Networks can be generated using __controlled parameters__ that govern structural features such as __density__, __reciprocity__, and __hub–spoke organization__.

## Repository Structure

```bash
environment.yml          # Environment configuration for conda.
requirements.txt         # Environment requirements for pip.
src/
    ├── Config.py                # Configuration parsing and validation
    ├── Structures.py            # Core data structures (nodes, arcs, commodities)
    ├── NetworkGenerator.py      # Network generator (hub-and-spoke, random, and emulation)
    ├── InstanceGenerator.py     # Instance and demand generator
    ├── main.py                  # Entry point for execution
    └── Config.txt               # Configuration file
data/
    ├── Networks/                # Default path for generated networks
        └── Benchmark/           # Benchmark networks used in the article
    └── Instances/               # Default path for generated instances
```

## Running the Generator
To generate networks and instances, run:
```
python src/main.py
```
The generator automatically reads parameters from [Config.txt](src/Config.txt).

By default, the generated networks are stored in [Networks/](data/Networks/), and the generated instances in [Instances/](data/Instances/). It can be changed with the parameters `defaultInstancePath` and `defaultNetworkPath`, respectively.

## Configuration File
All the parameters are read from the file [Config.txt](src/Config.txt). All the parameters marked as optional can be assigned the value `None` to be disabled.

The file is organized in three sections:
### General Parameters

| Parameter             | Type    | Description                                                   |
| --------------------- | ------- | ------------------------------------------------------------- |
| `defaultInstancePath` | Path    | Default path for reading/writing instances (`data/Instances`) |
| `defaultNetworkPath`  | Path    | Default path for reading/writing networks (`data/Networks`)   |
| `folder`              | str     | Optional subfolder for organizing runs                        |
| `networkNb`           | int > 0 | Number of networks to generate                                |
| `instanceNb`          | int > 0 | Number of instances to generate per network                   |
| `networkSeed`         | int     | Optional seed for reproducible network generation             |
| `demandSeed`          | int     | Optional seed for reproducible demand generation              |

### Network Generation Parameters

| Parameter                   | Type          | Description                                                              |
| --------------------------- | ------------- | ------------------------------------------------------------------------ |
| `networkEmulationPath`      | Path          | If provided, emulate an existing network instead of generating a new one |
| `networkEmulationTimeLimit` | float > 0     | Time limit for network emulation                                         |
| `randomGeneration`          | bool          | Whether to use random or structured hub-and-spoke generation             |
| `capacity`                  | float > 0     | Capacity of each arc                                                     |
| `bboxWidth`, `bboxHeight`   | float > 0     | Dimensions of the bounding box for arc distance                          |
| `targetNodeNb`              | int > 0       | Number of nodes                                                          |
| `targetArcNb`               | int > 0       | Arc budget (optional alternative to density)                             |
| `targetDensity`             | float ∈ [0,1] | Desired network density                                                  |
| `targetReciprocity`         | float ∈ [0,1] | Desired proportion of bidirectional arcs                                 |
| `decayRate`                 | float > 0     | Decay rate controlling how spread out clusters are                       |
| `hnRatio`                   | float ∈ (0,1] | Ratio of hub nodes to total nodes                                        |
| `priceRatio`                | float > 0     | Conversion rate of distance in kilometers to fixed cost.                 |
| `ufCostRatio`               | float > 0     | Ratio between unit and fixed costs                                       |
| `mode`                      | int ∈ {1,2,3,4}    | Transportation mode (optional): 1 (LTL), 2 (Liner), 3 (Rail), 4 (Express)      |
| `ltlRangeDensity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Density lower and upper bound of the LTL transportation mode networks. Default: (0.06,0.74).     |
| `ltlRangeReciprocity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Reciprocity lower and upper bound of the LTL transportation mode networks. Default: (0.71,1.0).     |
| `linerRangeDensity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Density lower and upper bound of the Liner transportation mode networks. Default: (0.02,0.82).     |
| `linerRangeReciprocity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Reciprocity lower and upper bound of the Liner transportation mode networks. Default: (0.58,1.0).     |
| `railRangeDensity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Density lower and upper bound of the Rail transportation mode networks. Default: (0.02,0.06).     |
| `railRangeReciprocity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Reciprocity lower and upper bound of the Rail transportation mode networks. Default: (1.0,1.0).     |
| `expressRangeDensity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Density lower and upper bound of the Express transportation mode networks. Default: (0.03,0.06).     |
| `expressRangeReciprocity`           | (float,float) ∈ [0,1]<sup>2</sup>  | Reciprocity lower and upper bound of the Express transportation mode networks. Default: (0.19,0.81).     |

### Demand Generation Parameters

| Parameter                                 | Type          | Description                                                                       |
| ----------------------------------------- | ------------- | --------------------------------------------------------------------------------- |
| `doStatic`                                | bool          | If true, generate SNDP; otherwise, generate SSNDP                                 |
| `commodityNb`                             | int > 0       | Number of commodities                                                             |
| `quantityToCapaMean`, `quantityToCapaDev` | float ∈ [0,1] | Mean and standard deviation of commodity size relative to arc capacity            |
| `sameRegionRatio`                         | float ∈ [0,1] | Ratio of commodities with origin-destinations lying in the same cluster           |
| `disparityRatio`                          | float ∈ [0,1] | Likelihood of uneven distribution of demand origins/destinations                  |
| `horizon`                                 | int > 0       | Planning horizon in number of days                                                |
| `discretization`                          | int > 0       | Number of homogeneous time periods in the planning horizon                        |
| `speed`                                   | float > 0     | Vehicle speed in kilometers per hour.                                             |
| `flexibilityMean`, `flexibilityDev`       | float ∈ [0,1] | Distribution of time flexibility relative to shortest path                        |
| `criticalTime`                            | int > 0       | Time rounding for available and due times. Must no exceed discretization value    |
| `distributionPattern`                     | list[float]   | Probability distribution of available times, size must be equal to discretization |
| `preProcessingSSNDP`                      | bool          | Whether to add preprocessing information (time windows) for SSNDP instances       |

The validity of all the parameters of the configuration file is checked before running the generation process and a ValueError
is reported to the user if incoherent values are given.

## Parameter Combinations
Any parameter in `Config.txt` can take multiple values (as a list).

All combinations across parameters are automatically enumerated.

Example:
```bash
targetDensity=[0.2,0.5,0.8]
targetReciprocity=0.5
```
Generates three sets of networks with the same reciprocity and varying densities. The same parameter validation process detailed before is applied to all combinations.

## Output Specification

### File Naming Convention
Each generated network or instance file includes encoded parameters in its name. Optional parameters not used are not part of the generated file name.

#### Networks
| Parameter           | Label |
| ------------------- | ----- |
| `decayRate`         | DR    |
| `hnRatio`           | A     |
| `ufCostRatio`       | UF    |
| `targetReciprocity` | R     |
| `targetDensity`     | D     |
| `targetNodeNb`      | N     |
| `networkSeed`       | S     |

#### (S)SNDP Instances
| Parameter            | Label |
| -------------------- | ----- |
| `commodityNb`        | C     |
| `quantityToCapaMean` | MCQ   |
| `quantityToCapaDev`  | DCQ   |
| `sameRegionRatio`    | SR    |
| `disparityRatio`     | DR    |
| `horizon`            | H     |
| `discretization`     | D     |
| `flexibilityMean`    | FM    |
| `flexibilityDev`     | FD    |
| `criticalTime`       | CT    |
| `demandSeed`         | S     |

Values in [0,1] are scaled by 100 and rounded.
Additional suffixes:
+ I: Index of the generated file (e.g., I0)
+ Prefix SNDP_ or SSNDP_ indicates the instance type

Examples of outputs names:
+ DR30_A1_UF5_R20_D5_N50_I0_S0: network with _decayRate_=30 ; _hnRatio_=0.01 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.2 ; _targetDensity_=0.05 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0
+ SNDP_MCQ10_DCQ50_C100_I0_DR30_A20_UF5_R50_D50_N50_I0_S0: SNDP instance with _quantityToCapaMean_=0.1 ; _quantityToCapaDev_=0.5 ; _targetCommodityNb_=100 ; _instanceIdx_=0 ; _decayRate_=30 ; _hnRatio_=0.2 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.5 ; _targetDensity_=0.5 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0
+ SSNDP_MCQ10_DCQ50_SR100_DR50_H24_FM50_FD17_CT5_C100_I0_DR30_A20_UF5_R50_D50_N50_I0_S0: SSNDP instance with _quantityToCapaMean_=0.1 ; _quantityToCapaDev_=0.5 ; _sameRegionRatio_=1.0 ; _disparityRatio_=0.5 ; _horizon_=24 ; _flexibilityMean_=0.5 ; _flexibilityDev_=0.17 ; _criticalTime_=5 ; _targetCommodityNb_=100 ; _instanceIdx_=0 ; _decayRate_=30 ; _hnRatio_=0.2 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.5 ; _targetDensity_=0.5 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0

#### Output files structure
The file structure of the outputs of the generator follows the structure:
```
NODES,<number of nodes generated>
...
node id, cluster id, x, y
...
ARCS,<number of arcs generated>
...
arc id, origin id, destination id, unit cost, fixed cost, capacity, distance
...
COMMODITIES,<number of commodities generated>
...
commodity id, origin id, destination id, quantity, available time, due time
...
horizon=<length of the planning horizon>
discretizatio=<discretization>
distribution_pattern=<distributionPattern>
COMMODITY_NODE_TIMEWINDOWS,<number of per commodity, per node, time windows>
...
time window id, commodity id, node id, lower bound, upper bound
...
COMMODITY_ARC_TIMEWINDOWS,<number of per commodity, per arc, time windows>
...
time window id, commodity id, arc id, lower bound, upper bound
...
```
Some remarks:
+ The x and y coordinates are not given if no bounding box width and height was specified.
+ The arc distance, in network files, is an euclidean distance represented by float numbers. It is not given in SNDP instance files. It is given as a number of time period in SSNDP instance files.
+ The horizon is only given for SSNDP instance files.
+ The distribution pattern is only given for SSNDP instance files where the associated parameter was specified.
+ The two lists of time windows (for arcs and nodes) are given only for SSNDP instance files when the parameter _preProcessingSSNDP_=True.

## Replicating

This project is designed to be fully reproducible using either `pip` or `conda`. Use either `requirements.txt` or `environment.yml` to do so.

## Ongoing Development
This code is being developed on an on-going basis at the author's [GitHub site](https://github.com/LBonnet159/SSNDP-Realistic-Instance-Generator).

If you are interested in specific areas of the generator, or in adding new functionalities to it, we encourage you to contact the authors and discuss the desired feature request.

## Support
For support in using this software, submit an [issue](https://github.com/LBonnet159/SSNDP-Realistic-Instance-Generator/issues/new).
