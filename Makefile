.PHONY: docs

docs:
	(cd docs && make html)

lint:
	mypy lcs

test: lint
	py.test -n 4 --pep8 -m pep8
	py.test -n 4 --cov=lcs tests/

notebook:
	jupyter notebook

execute_notebooks:
	find . -name '*Lake.ipynb' -exec jupyter nbconvert --execute {} --inplace --debug --ExecutePreprocessor.timeout=6000 \;
