.PHONY: docs

docs:
	(cd docs && make html)

test:
	py.test -n 4 --cov=lcs tests/
