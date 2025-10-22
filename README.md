# SSNDP-Realistic-Instance-Generator
This archive is distributed under the MIT license.

The software and data in this repository were used in the research reported in the article "A Complex Network Analysis Approach for Generating Realistic Instances of the Scheduled Service Network Design Problem" by Louis Bonnet, Simon Belieres, Mike Hewitt, and Sandra Ulrich Ngueveu.

## Description
The Service Network Design Problem (SNDP), and its timed variant the Scheduled SNDP (SSNDP), are challenging optimization problems. This code generates instances for these problems, consisting of a list of nodes, arcs, and commodities.

The [src/](src/) directory contains the generator's source python (.py) code and the configuration [Config.txt](src/Config.txt) file. The [data/](data/) directory contains folders [Networks/](data/Networks/) and [Instances/](data/Instances). Networks and instances are generated in and read from these respective folders by default. The 180 networks presented in the article "A Complex Network Analysis Approach for Generating Realistic Instances of the Scheduled Service Network Design Problem" are found in the directory [SetBenchmarks/](data/Networks/SetBenchmarks/).

## Generating instances
In order to run the generator, type:
```
python src/main.py
```

### Inputs
All the parameters of the generator are read from file [Config.txt](src/Config.txt). The range, type, and usage of the parameters are:

(general configuration parameters)
+ _defaultInstancePath_ $\in$ absolute or relative path, default=data/Instances ; path to read/write (S)SNDP instances
+ _defaultNetworkPath_ $\in$ absolute or relative path, default=data/Networks ; path to read/write networks
+ _folder_ $\in$ string ; folder name in which to read/write instances/networks at related default path
+ _networkNb_ $\in \mathbb{N}_{>0}$ ; number of networks to generate
+ _instanceNb_ $\in \mathbb{N}_{>0}$ ; number of instances to generate per network
+ _networkSeed_ $\in \mathbb{N}_{>0}$ ; (optional) seed to use when generating networks
+ _demandSeed_ $\in \mathbb{N}_{>0}$ ; (optional) seed to use when generating (S)SNDP instances

(network generation parameters)
+ _networkEmulationPath_ $\in$ absolute path ; absolute path to the network to emulate
+ _networkEmulationTimeLimit_ $\in \mathbb{R}_{>0}$ ; time limit of the network emulation
+ _randomGeneration_ $\in \mathbb{B}$ ; whether to generate the network randomly or in a structured hub-and-spoke manner
+ _capacity_ $\in \mathbb{R}_{>0}$ ; capacity of the arcs of the generated network
+ _bboxWidth_ $\in \mathbb{R}_{>0}$ ; width of the bounding box of the generated network
+ _bboxHeight_ $\in \mathbb{R}_{>0}$ ; height of the bounding box of the generated network
+ _targetNodeNb_ $\in \mathbb{N}_{>0}$ ; number of nodes to generate
+ _targetArcNb_ $\in \mathbb{N}_{>0}$ ; arc budget
+ _targetDensity_ $\in [0,1]$ ; density of the network to generate, derived to an arc budget
+ _targetReciprocity_ $\in [0,1]$ ; reciprocity to reach on average
+ _decayRate_ $\in \mathbb{R}_{>0}$ ; decay rate of the clustered nodes generation
+ _hnRatio_ $\in ]0,1]$ ; ratio between the number of hub nodes and the total number of nodes
+ _ufCostRatio_ $\in \mathbb{R}_{>0}$ ; ratio between the unit and fixed costs

(demand generation parameters)
+ _doStatic_ $\in \mathbb{B}$ ; whether to produce a SNDP or a SSNDP instance
+ _commodityNb_ $\in \mathbb{N}_{>0}$ ; number of commodities to generate
+ _quantityToCapaMean_ $\in [0,1]$ ; mean of the truncated normal distribution of the commodities quantity relative to the arcs capacity
+ _quantityToCapaDev_ $\in [0,1]$ ; standard deviation of the truncated normal distribution of the commodities quantity relative to the arcs capacity
+ _sameRegionRatio_ $\in [0,1]$ ; (optional) ratio of commodities whose origin and destination lie in the same cluster
+ _disparityRatio_ $\in [0,1]$ ; (optional) likelihood of origins and destinations to be spread evenly across clusters or not (0 = even spread, 1 = uneven spread) 
+ _horizon_ $\in \mathbb{N}_{>0}$ ; length of the planning horizon
+ _flexibilityMean_ $\in  [0,1]$ ; mean of the truncated normal distribution of the flexibility, relative to the shortest path length
+ _flexibilityDev_ $\in [0,1]$ ; standard deviation of the truncated normal distribution of the flexibility, relative to the shortest path length
+ _criticalTime_ $\in \mathbb{N}_{>0}$ ; (optional) critical time to round down considered available times, and round up considered due times
+ _distributionPattern_ $\in [0,1]^{\textit{horizon}}$ ; (optional) probability distribution of the available times over the planning horizon
+ + _preProcessingSSNDP_ $\in \mathbb{B}$ ; add pre-processings to generated SSNDP instances

The range of the parameters is checked before the generation and an error is reported to the user if incoherent values are given.

In order to produce networks and instances with different inputs more easily, values given to parameters in the file [Config.txt](src/Config.txt) can
be written as lists. For instance, if _targetDensity_=[0.2,0.5,0.8], and all other parameters have a single value (e.g. , _targetReciprocity_=0.5), three sets of configuration paremeters
will be used.

### Output format

We detail the format of the outputs of the generator.

#### Ouputs name label
Generated networks and instances are named according to some of the parameters used during the generation (the one that influence the generation procedure). We list now the label of each parameter when naming an output. 

First, in the case of the network generation:
+ _decayRate_: DR
+ _hubNodeRatio_: A
+ _ufCostRatio_: UF
+ _targetReciprocity_: R
+ _targetDensity_: D
+ _targetNodeNb_: N
+ _networkSeed_: S

Second, in the case of the (S)SNDP instance generation:
+ _quantityToCapaMean_: MCQ
+ _quantityToCapaDev_: DCQ
+ _sameRegionRatio_: SR
+ _disparityRatio_: DR
+ _targetCommodityNb_: N
+ _horizon_: H
+ _flexibilityMean_: FM
+ _flexibilityDev_: FD
+ _criticalTime_: CT
+ _demandSeed_: S

The labels are written with the value of the associated parameter. For parameters in the range [0,1], the value is multiplied by 100 and rounded. To differentiate networks or instances generated at the same time with the same input parameters, we also add label I, associated with the index of the generation, _networkIdx_ for the network, _instanceIdx_ for the instance. Finally, SNDP instances have the prefix SNDP, while SSNDP instances have the prefix SSNDP.

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
