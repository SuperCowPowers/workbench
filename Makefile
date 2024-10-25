.PHONY: help clean clean-pyc clean-build list 

help:
	@echo "clean - remove ALL build/python/misc artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python Complied file artifacts"
	@echo "clean-misc - remove misc artifacts"

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

