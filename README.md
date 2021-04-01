# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/ParrotPrediction/pyalcs.svg?branch=master)](https://travis-ci.org/ParrotPrediction/pyalcs) [![Documentation Status](https://readthedocs.org/projects/pyalcs/badge/?version=latest)](https://pyalcs.readthedocs.io/en/latest/?badge=latest)

The main advantage of *Learning Classifier Systems* with respect to other RL techniques it to afford generalization capabilities. This makes it possible to aggregate several situations within a common description so that the representation of the problem gets smaller.

## Agents

### ACS
Introduced by _Stolzmann_ in 1997 originally intended to simulate and evaluate Hoffmann's learning theory of anticipations.
- LCS framework with explicit representation of anticipations
- directed anticipatory learning process

### ACS2
Added modifications:
- start with initially empty population of classifiers that are created by covering mechanism,
- genetic generalization mechanism
- population includes C-A-E triples that anticipate no change in the environment (ACS by default assumes no changes),
- after executing an action modification are applied to all action set [A],
- classifier has an extra property of "immediate reward".

### YACS
- Different heuristics
- Does not generalize wrt. payoff prediction (therefore classifiers itself don't store the information about the reward)

### MACS 
- Extends ACS, ACS2, YACS by the possibility to detect interactions between attributes (because they consider each situation as an unsecable whole - new value of the attribute may only depend upon the previous value of the same attribute).
- A _"don't know (?)"_ symbol is used in effect part instead of _"don't change (#)"_.  

References
- _["Combining latent learning with dynamic programming in the modular anticipatory classifier system"](https://www.sciencedirect.com/science/article/pii/S0377221703006611)_ by P. Gérard, JA. Meyer, O. Sigaud 

## Documentation
Documentation is available [here](https://pyalcs.readthedocs.io).

### Citation
If you want to use the library in your project please cite the following:

    @inbook{
        title = "Integrating Anticipatory Classifier Systems with OpenAI Gym",
        keywords = "Aniticipatory Learning Classifier Systems, OpenAI Gym",
        author = "Norbert Kozlowski, Olgierd Unold",
        year = "2018",
        doi = "10.1145/3205651.3208241",
        isbn = "978-1-4503-5764-7/18/07",
        booktitle = "Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '18)",
        publisher = "Association for Computing Machinery",
    }

Prior to PR please execute to check if standards are holding:

    make test

## See also
- [Dedicated OpenAI Gym environments](https://github.com/ParrotPrediction/openai-envs)
- [Examples of integration and interactive notebooks](https://github.com/ParrotPrediction/pyalcs-experiments)
