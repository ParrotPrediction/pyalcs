# ACS2
[![Build Status](https://travis-ci.org/khozzy/ACS2.svg?branch=master)](https://travis-ci.org/khozzy/ACS2)

## Contribution
If you want to contribute, please execute tests beforehand:

    python3 -m unittest discover
    
and check PEP8 compliance:

    find . -name \*.py -exec pep8 --ignore=E129 {} +
    
## Thoughts

- mind using `collections.deque` for storing past data,
- `heapq` for finding n largest/smallest values in various iterables (matching classifiers)
