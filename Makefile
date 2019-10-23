.PHONY: docs

docs:
	(cd docs && make html)

lint:
	mypy lcs

test: lint
	py.test -n 4 --pep8 -m pep8
	py.test -n 4 --cov=lcs tests/

notebook:
	jupyter lab --notebook-dir notebooks/

execute_notebooks:
#	papermill notebooks/rACS_Corridor.ipynb docs/source/notebooks/rACS_Corridor.ipynb
	papermill notebooks/FrozenLake.ipynb docs/source/notebooks/FrozenLake.ipynb
	papermill notebooks/Maze.ipynb docs/source/notebooks/Maze.ipynb
	papermill notebooks/ACS2.ipynb docs/source/notebooks/ACS2.ipynb
