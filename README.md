# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/khozzy/ACS.svg?branch=master)](https://travis-ci.org/khozzy/ACS)


ALCS are is an extension to basic LCS compromising the notation of anticipations. Doing that the systems predominantly are able to **anticipate perceptual consequences of actions** independent of a reinforcement predictions.
 
 ALCS are able to form complete anticipatory representation (build environment model) which allows faster
 and more intelligent adaptation of behaviour or problem classification.

## Abstraction
The framework abstraction layer is divided into two components - agents and environments.

### Agents
Currently implemented
#### [ACS2](acs/agent/acs2/ACS2.py)
ACS2 is derived from the original ACS framework. The most important change is that it embodies genetic generalization mechanism. Implementation based on *"An Algorithmic Description of ACS2"* by Martin V. Butz and Wolfgang Stolzmann.

### Environments
At this moment it is possible to put an agent into a maze defined in an external file.

## Examples
To run ACS2 algorithm from `examples/` directory type:

    make example_acs2

## Contribution
Prior to PR please execute:

    make install_deps
    
and check if standards are holding:

    make test
    make pep8

#### Dev thoughts
- `any` method might be nice,
- integrate with CodeClimate
