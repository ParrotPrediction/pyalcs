# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/khozzy/pyalcs.svg?branch=master)](https://travis-ci.org/khozzy/pyalcs)


ALCS are is an extension to basic LCS compromising the notation of anticipations. Doing that the systems predominantly are able to **anticipate perceptual consequences of actions** independent of a reinforcement predictions.
 
 ALCS are able to form complete anticipatory representation (build environment model) which allows faster
 and more intelligent adaptation of behaviour or problem classification.


## Deep dive
Before working with code please install few required dependencies (code is running on Python 3)

    make install_deps

Most of the analysis is done via Jupyter notebooks, you can start one with

    make notebook

On your browser proceed to `localhost:8888`, from there you can navigate to `examples/notebooks`.

Files worth checking:

- *Description of ACS2.ipynb* - mathematical explanation of the algorithm,
- *ACS2 in Maze.ipynb* - how the algorithm is performing in maze environment.

Other example of running the algorithm (not from the notebook) which might be quite handy for debugging.

    python3 examples/run_acs2.py

## Original code
The original author's code is located in `assets/original` directory.

However it was written in 2001 when C++ was quite different than now. For that reason a slightly changed version (syntax) working on nowadays compilers can be found in `assets/ACS2`.

To compile the sources type (inside `assets/ACS2`):

    make

And to run it:

    ./acs2++.out <environment>

For example:

    ./acs2++.out Envs/Maze4.txt

## Framework abstraction
The framework abstraction layer is divided into two components - agents and environments.

### Agents
Currently implemented
#### [ACS2](acs/agent/acs2/ACS2.py)
ACS2 is derived from the original ACS framework. The most important change is that it embodies genetic generalization mechanism. Implementation based on *"An Algorithmic Description of ACS2"* by Martin V. Butz and Wolfgang Stolzmann.

### Environments
At this moment it is possible to put an agent into a maze, however there plans for integration with OpenAI Gym.

#### Maze
Environment where agent (randomly placed) is supposed to find reward.

## Contribution
Prior to PR please execute to check if standards are holding:

    make test
    make coverage
    make pep8
