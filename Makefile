install_deps:
	pip3 install -r requirements.txt --ignore-installed six
test:
	python3 -m unittest -v tests
coverage:
	coverage run -m unittest discover -v tests
	coverage report -m
	rm .coverage
pep8:
	find . -name \*.py -exec pep8 --ignore=E129,E222,E402 {} +
notebook:
	jupyter notebook
