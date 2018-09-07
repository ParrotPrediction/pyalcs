.PHONY: docs

docs:
	(cd docs && make html)

lint: docs
	mypy lcs

test: lint
	py.test -n 4 --pep8 -m pep8
	py.test -n 4 --cov=lcs tests/

notebook:
	jupyter notebook
