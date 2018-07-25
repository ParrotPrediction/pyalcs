install_deps:
	pip install -r requirements.txt --ignore-installed six
test:
	py.test --pep8 -m pep8
	py.test --cov=lcs tests/
notebook:
	jupyter notebook
