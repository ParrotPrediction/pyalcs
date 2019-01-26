.PHONY: docs

docs:
	(cd docs && make html)

type_check:
	mypy lcs

test: type_check
	py.test -n 4 --pep8 -m pep8
	py.test -n 4 --cov=lcs tests/

notebook:
	jupyter notebook

execute_notebooks:
	find . -name '*Lake.ipynb' -exec jupyter nbconvert --execute {} --inplace --debug --ExecutePreprocessor.timeout=6000 \;
