# SSNDP-Realistic-Instance-Generator
This archive is distributed under the MIT license.

The software and data in this repository were used in the research reported in the article "A Complex Network Analysis Approach for Generating Realistic Instances of the Scheduled Service Network Design Problem" by Louis Bonnet, Simon Belieres, Mike Hewitt, and Sandra Ulrich Ngueveu.

## Description
The Service Network Design Problem (SNDP), and its timed variant the Scheduled SNDP (SSNDP), are challenging optimization problems. This code generates instances for these problems, consisting of a list of nodes, arcs, and commodities.

The [src/](src/) directory contains the generator's source python (.py) code and the configuration [Config.txt](src/Config.txt) file. The [data/](data/) directory contains folders [Networks/](data/Networks/) and [Instances/](data/Instances). Networks and instances are generated in and read from these respective folders by default. The 180 networks presented in the article "A Complex Network Analysis Approach for Generating Realistic Instances of the Scheduled Service Network Design Problem" are found in directory [SetBenchmarks/](data/Networks/SetBenchmarks/).

## Generating instances
In order to run the generator, type:
```
python src/main.py
```

All the parameters of the generator are read from file [Config.txt](src/Config.txt).

## How to contribute
Thank you for considering contributing to our project! To report bugs and ask questions, please refer to the official issue tracker. You can also address a problem by (1) forking the project, (2) correcting the bug / adding a feature, and (3) generating a pull request. However, we recommend that you first contact the authors and discuss the desired feature request.

You are also very welcome if you want to upload in our repository the instances you generated with our generator and used in your work! In this way, other researchers can use the same set of instances as benchmark, without having to re-generate such instances using the parameters you used. If this is the case, please send us an email and we will take care of handling your request.
