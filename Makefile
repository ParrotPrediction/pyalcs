test:
	py.test --pep8 -m pep8
	py.test --cov=lcs tests/
notebook:
	jupyter notebook
