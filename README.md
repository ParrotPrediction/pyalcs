# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/ParrotPrediction/pyalcs.svg?branch=master)](https://travis-ci.org/ParrotPrediction/pyalcs)

ALCS are is an extension to basic LCS compromising the notation of anticipations. Doing that the systems predominantly are able to **anticipate perceptual consequences of actions** independent of a reinforcement predictions.
 
 ALCS are able to form complete anticipatory representation (build environment model) which allows faster
 and more intelligent adaptation of behaviour or problem classification.


## Deep dive
Before working with code please install few required dependencies (code is running on Python 3)

    make install_deps

## Original code
The original author's code is located in `assets/original` directory.

However it was written in 2001 when C++ was quite different than now. For that reason a slightly changed version (syntax) working on nowadays compilers can be found in `assets/ACS2`.

To compile the sources type (inside `assets/ACS2`):

    make

And to run it:

    ./acs2++.out <environment>

For example:

    ./acs2++.out Envs/Maze4.txt

### Agents

#### [ACS2](alcs/acs2/ACS2.py)
ACS2 is derived from the original ACS framework. The most important change is that it embodies genetic generalization mechanism. Implementation based on *"An Algorithmic Description of ACS2"* by Martin V. Butz and Wolfgang Stolzmann.

## Contribution
Prior to PR please execute to check if standards are holding:

    make test
    make coverage
    make pep8
