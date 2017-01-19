# Anticipatory Classifier Systems in Python
[![Build Status](https://travis-ci.org/khozzy/ACS.svg?branch=master)](https://travis-ci.org/khozzy/ACS)

Repository containing code implementation for various *Anticipatory Classifier Systems*.

## Abstraction
The abstraction layer is divided into two components - agents and environments.

### Agents
Currently implemented agents: [ACS2](acs/agent/acs2/ACS2.py).

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
- `any` method might be nice
