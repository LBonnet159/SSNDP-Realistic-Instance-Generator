# SSNDP-Realistic-Instance-Generator
This archive is distributed under the MIT license.

The software and data in this repository were used in the research reported in the article "A Complex Network Analysis Approach for Generating Realistic Instances of the Scheduled Service Network Design Problem" by Louis Bonnet, Simon Belieres, Mike Hewitt, and Sandra Ulrich Ngueveu.

## Overview
The Service Network Design Problem (SNDP), and its timed variant the Scheduled SNDP (SSNDP), are challenging optimization problems arising in freight and transportation systems. This software generates realistic instances for these problems based on the literature on Complex Networks.

Each instance, either of the SNDP or the SSNDP, consists of:
+ A __directed network__ with nodes and arcs
+ A __set of commodities__

Networks and instances can be generated using __controlled parameters__ that govern structural features such as __density__, __reciprocity__, and __hub–spoke organization__.

## Repository Structure

```bash
src/
    ├── Config.py                # Configuration parsing and validation
    ├── Structures.py            # Core data structures (nodes, arcs, commodities)
    ├── NetworkGenerator.py      # Network generator (hub-and-spoke, random, and emulation)
    ├── InstanceGenerator.py     # Instance and demand generator
    ├── main.py                  # Entry point for execution
    └── Config.txt               # Configuration file

data/
    ├── Networks/                # Default path for generated networks
    ├── Instances/               # Default path for generated instances
    └── Networks/SetBenchmarks/  # Benchmark networks used in the article
```

## Running the Generator
To generate networks and instances, run:
```
python src/main.py
```
The generator automatically reads parameters from [Config.txt](src/Config.txt).

By default, generated networks are stored in [Networks/](data/Networks/), and generated instances in [Instances/](data/Instances/).

## Configuration File
All parameters are read from file [Config.txt](src/Config.txt).

The file is organized in three sections:
### General Parameters

| Parameter             | Type    | Description                                                   |
| --------------------- | ------- | ------------------------------------------------------------- |
| `defaultInstancePath` | Path    | Default path for reading/writing instances (`data/Instances`) |
| `defaultNetworkPath`  | Path    | Default path for reading/writing networks (`data/Networks`)   |
| `folder`              | str     | Optional subfolder for organizing runs                        |
| `networkNb`           | int > 0 | Number of networks to generate                                |
| `instanceNb`          | int > 0 | Number of instances to generate per network                   |
| `networkSeed`         | int > 0 | Optional seed for reproducible network generation             |
| `demandSeed`          | int > 0 | Optional seed for reproducible demand generation              |

### Network Generation Parameters

| Parameter                   | Type          | Description                                                              |
| --------------------------- | ------------- | ------------------------------------------------------------------------ |
| `networkEmulationPath`      | Path          | If provided, emulate an existing network instead of generating a new one |
| `networkEmulationTimeLimit` | float > 0     | Time limit for network emulation                                         |
| `randomGeneration`          | bool          | Whether to use random or structured hub-and-spoke generation             |
| `capacity`                  | float > 0     | Capacity of each arc                                                     |
| `bboxWidth`, `bboxHeight`   | float > 0     | Dimensions of the bounding box for arc distance                        |
| `targetNodeNb`              | int > 0       | Number of nodes                                                          |
| `targetArcNb`               | int > 0       | Arc budget (optional alternative to density)                             |
| `targetDensity`             | float ∈ [0,1] | Desired network density                                                  |
| `targetReciprocity`         | float ∈ [0,1] | Desired proportion of bidirectional arcs                                         |
| `decayRate`                 | float > 0     | Decay rate controlling how spread out clusters are                               |
| `hnRatio`                   | float ∈ (0,1] | Ratio of hub nodes to total nodes                                        |
| `ufCostRatio`               | float > 0     | Ratio between unit and fixed costs                                       |

### Demand Generation Parameters

| Parameter                                 | Type          | Description                                                                 |
| ----------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| `doStatic`                                | bool          | If true, generate SNDP; otherwise, generate SSNDP                           |
| `commodityNb`                             | int > 0       | Number of commodities                                                       |
| `quantityToCapaMean`, `quantityToCapaDev` | float ∈ [0,1] | Mean and standard deviation of commodity size relative to arc capacity      |
| `sameRegionRatio`                         | float ∈ [0,1] | Ratio of commodities with origin-destinations lying in the same cluster     |
| `disparityRatio`                          | float ∈ [0,1] | Likelihood of uneven distribution of demand origins/destinations            |
| `horizon`                                 | int > 0       | Planning horizon (time periods)                                             |
| `flexibilityMean`, `flexibilityDev`       | float ∈ [0,1] | Distribution of time flexibility relative to shortest path                  |
| `criticalTime`                            | int > 0       | Time rounding for available and due times                                   |
| `distributionPattern`                     | list[float]   | Probability distribution of available times, size must be equal to horizon  |
| `preProcessingSSNDP`                      | bool          | Whether to add preprocessing information (time windows) for SSNDP instances |

The range of the parameters is checked before the generation and an error is reported to the user if incoherent values are given.

In order to produce networks and instances with different inputs more easily, values given to parameters in the file [Config.txt](src/Config.txt) can
be written as lists. For instance, if _targetDensity_=[0.2,0.5,0.8], and all other parameters have a single value (e.g. , _targetReciprocity_=0.5), three sets of configuration paremeters
will be used.

## Parameter Combinations
Any parameter in `Config.txt` can take multiple values (as a list).

All combinations across parameters are automatically enumerated.

Example:
```bash
targetDensity=[0.2,0.5,0.8]
targetReciprocity=0.5
```
Generates three sets of networks with the same reciprocity and varying densities.

The software validates all parameter values before execution. An error is returned to the user if incoherent values are detected.

## Output Specification

### File Naming Convention
Each generated network or instance file includes encoded parameters in its name.

#### Networks
| Parameter           | Label |
| ------------------- | ----- |
| `decayRate`         | DR    |
| `hubNodeRatio`      | A     |
| `ufCostRatio`       | UF    |
| `targetReciprocity` | R     |
| `targetDensity`     | D     |
| `targetNodeNb`      | N     |
| `networkSeed`       | S     |

#### (S)SNDP Instances
| Parameter            | Label |
| -------------------- | ----- |
| `quantityToCapaMean` | MCQ   |
| `quantityToCapaDev`  | DCQ   |
| `sameRegionRatio`    | SR    |
| `disparityRatio`     | DR    |
| `horizon`            | H     |
| `flexibilityMean`    | FM    |
| `flexibilityDev`     | FD    |
| `criticalTime`       | CT    |
| `demandSeed`         | S     |

Values in [0,1] are scaled by 100 and rounded.
Additional suffixes:
+ I: Index of the generated file (e.g., I0)
+ Prefix SNDP_ or SSNDP_ indicates the instance type

Examples of outputs names:
+ DR30_A1_UF5_R20_D5_N50_I0_S0: network with _decayRate_=30 ; _hubNodeRatio_=0.01 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.2 ; _targetDensity_=0.05 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0
+ SNDP_MCQ10_DCQ50_C100_I0_DR30_A20_UF5_R50_D50_N50_I0_S0: SNDP instance with _quantityToCapaMean_=0.1 ; _quantityToCapaDev_=0.5 ; _targetCommodityNb_=100 ; _instanceIdx_=0 ; _decayRate_=30 ; _hubNodeRatio_=0.2 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.5 ; _targetDensity_=0.5 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0
+ SSNDP_MCQ10_DCQ50_SR100_DR50_H24_FM50_FD17_CT5_C100_I0_DR30_A20_UF5_R50_D50_N50_I0_S0: SSNDP instance with _quantityToCapaMean_=0.1 ; _quantityToCapaDev_=0.5 ; _sameRegionRatio_=1.0 ; _disparityRatio_=0.5 ; _horizon_=24 ; _flexibilityMean_=0.5 ; _flexibilityDev_=0.17 ; _criticalTime_=5 ; _targetCommodityNb_=100 ; _instanceIdx_=0 ; _decayRate_=30 ; _hubNodeRatio_=0.2 ; _ufCostRatio_=0.05 ; _targetReciprocity_=0.5 ; _targetDensity_=0.5 ; _targetNodeNb_=50 ; _networkIdx_=0 ; _networkSeed_=0

#### Output files structure
The file structure of the outputs of the generator follows the structure:
```
NODES,\<number of nodes generated\>
...
node id, cluster id, x, y
...
ARCS,\<number of arcs generated\>
...
arc id, origin id, destination id, unit cost, fixed cost, capacity, distance
...
COMMODITIES,\<number of commodities generated\>
...
commodity id, origin id, destination id, quantity, available time, due time
...
horizon=\<length of the planning horizon\>
distribution_pattern=\<_distributionPattern_\>
COMMODITY_NODE_TIMEWINDOWS,\<number of per commodity, per node, time windows\>
...
time window id, commodity id, node id, lower bound, upper bound
...
COMMODITY_ARC_TIMEWINDOWS,\<number of per commodity, per arc, time windows\>
...
time window id, commodity id, arc id, lower bound, upper bound
...
```
Some remarks:
+ The cluster id value is not given in the case of a random network.
+ The x and y coordinates are not given if no bounding box width and heigth was specified.
+ The arc distance, in network files, is an euclidean distance represented by float numbers. It is not given in SNDP instance files. It is given as a number of time period in SSNDP instance files.
+ The horizon is only given for SSNDP instance files.
+ The distribution pattern is only given for SSNDP instance files where the associated parameter was specified.
+ The two lists of time windows (for arcs and nodes) are given only for SSNDP instance files when the parameter _preProcessingSSNDP_=True.

## How to contribute
Thank you for considering contributing to our project! To report bugs and ask questions, please refer to the issue tracker. You can also address a problem by (1) forking the project, (2) correcting the bug / adding a feature, and (3) generating a pull request. However, we recommend that you first contact the authors and discuss the desired feature request.

You are also very welcome if you want to upload in our repository the instances you generated with our generator and used in your work! In this way, other researchers can use the same set of instances as benchmark, without having to re-generate such instances using the parameters you used. If this is the case, please send us an email.
