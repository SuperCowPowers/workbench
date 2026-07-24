.PHONY: help clean clean-pyc clean-build list

# Prune virtualenv and git internals so cleans never touch installed packages
PRUNE := \( -path ./.venv -o -path ./.git \) -prune -o

help:
	@echo "clean - remove ALL build/python/misc artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python Complied file artifacts"
	@echo "clean-misc - remove misc artifacts"

clean: clean-build clean-pyc clean-misc

clean-build:
	find . $(PRUNE) -name 'build' -exec rm -rf {} +
	find . $(PRUNE) -name '_build' -exec rm -rf {} +
	find . $(PRUNE) -name 'dist' -exec rm -rf {} +
	find . $(PRUNE) -name '*.egg-info' -exec rm -rf {} +
	find . $(PRUNE) -name '*.eggs' -exec rm -rf {} +
	find . $(PRUNE) -name '*.tar.gz' -exec rm -rf {} +
	find . $(PRUNE) -name '.tox' -exec rm -rf {} +
	find . $(PRUNE) -name '.coverage' -exec rm -rf {} +
	find . $(PRUNE) -name '.cache' -exec rm -rf {} +
	find . $(PRUNE) -name '__pycache__' -exec rm -rf {} +
	find . $(PRUNE) -name 'cdk.out' -exec rm -rf {} +

clean-pyc:
	find . $(PRUNE) -name '*.pyc' -exec rm -rf {} +
	find . $(PRUNE) -name '*.pyo' -exec rm -rf {} +
	find . $(PRUNE) -name '*~' -exec rm -rf {} +

clean-misc:
	find . $(PRUNE) -name '.ipynb_checkpoints' -exec rm -rf {} +

