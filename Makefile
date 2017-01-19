install_deps:
	pip install pep8
test:
	python3 -m unittest -v tests
pep8:
	find . -name \*.py -exec pep8 --ignore=E129,E402 {} +
run_acs2:
	python3 examples/run_acs2.py
