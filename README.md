# Anticipatory Learning Classifier Systems in Python
Repository containing code implementation for various *Anticipatory Learning Classifier Systems* (ALCS).

[![Build Status](https://travis-ci.org/ParrotPrediction/pyalcs.svg?branch=master)](https://travis-ci.org/ParrotPrediction/pyalcs)

ALCS are is an extension to basic LCS compromising the notation of anticipations. Doing that the systems predominantly are able to **anticipate perceptual consequences of actions** independent of a reinforcement predictions.
 
 ALCS are able to form complete anticipatory representation (build environment model) which allows faster and more intelligent adaptation of behaviour or problem classification.

## Deep dive

### Project setup

It is recommended to use a **Python 3.7** environment for development (virtualenv/virtualenvwrapper/conda).

#### Conda (recommended)
Having a Conda distribution (i.e. Anacoda, Minicoda etc) create environment like:

    conda create --name pyalcs python=3.7

Then activate it with:

    source activate pyalcs

#### Virtualenv
Create dedicated environment [virtualenv](https://virtualenv.pypa.io/) (only needs to be run once)

    virtualenv venv

Then, every time you want to work on a project from the shell, type the following to activate the virtual environment:

    source venv/bin/activate


#### Dependencies
Before working with code please install few required dependencies (code is running on Python 3). Make sure Swig binary is installed (required to compile OpenAI Gym environments) - just look at [TravisCI build file](.travis.yml).

### Launching example integrations

I assuming you are inside the virtual environment created before.
In order to run the integrations from the console you need to
specify Python PATH to use the currently checked-out version
of `alcs` library:

    PYTHONPATH=/Users/khozzy/Projects/pyalcs python examples/maze/acs2_in_maze.py

PyCharm IDE appends the path automatically.

### Interpreting statistics

After running an example integration, say `acs2_in_maze.py`, here's what the
output tells you:

`agent_stats`

See `lcs.agents.acs2.ACS2`

 * `population`: number of classifiers in the population
 * `numerosity`: sum of numerosities of all classifiers in the population
 * `reliable`: number of reliable classifiers in the population
 * `fitness`: average classifier fitness in the population
 * `trial`: trial number
 * `steps`: number of steps in this trial
 * `total_steps`: number of steps in all trials so far

`env_stats`

There are currently no environment statistics for maze environment.

`performance_stats`

 * `knowledge`: As defined in
   `examples.acs2.maze.utils.calculate_performance()`:
   If any of the reliable classifiers successfully predicts a transition, we
   say that the transition is anticipated correctly.  This is a percentage of
   correctly anticipated transitions among all possible transitions.

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
