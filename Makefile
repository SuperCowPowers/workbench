.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"

clean: clean-build clean-pyc clean-misc

clean-build:
	find . -name 'build' -exec rm -rf {} +
	find . -name '_build' -exec rm -rf {} +
	find . -name 'dist' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.eggs' -exec rm -rf {} +
	find . -name '*.tar.gz' -exec rm -rf {} +
	find . -name '.tox' -exec rm -rf {} +
	find . -name '.coverage' -exec rm -rf {} +
	find . -name '.cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +

clean-misc:
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +

lint:
	flake8 src/sageworks tests

test:
	tox

test-all:
	tox

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

sdist: clean
	python setup.py sdist upload
