#!/usr/bin/env python
"""Setup.py for SageWorks: AWS ML Pipeline Workbench"""

from setuptools import setup, find_namespace_packages

# Readme
with open("README.md", "r") as f:
    readme = f.read()

# Requirements
with open("requirements.txt", "r") as f:
    install_requires = f.read().strip().split("\n")

# Extra requirements
extras_require = {
    "ml-tools": [
        "shap>=0.43.0",
        "networkx>=3.2",
    ],
    "chem": ["rdkit>=2023.9.1", "mordredcommunity>=2.0"],
    "ui": [
        "plotly>=5.18.0",
        "dash>=2.16.1",
        "dash-bootstrap-components>=1.5.0",
        "dash-bootstrap-templates==1.1.1",
        "dash_ag_grid",
        "tabulate>=0.9.0",
    ],
    "dev": ["pytest", "pytest-sugar", "coverage", "pytest-cov", "flake8", "black"],
    "all": [
        "shap>=0.43.0",
        "networkx>=3.2",
        "rdkit>=2023.9.1",
        "mordredcommunity>=2.0",
        "plotly>=5.18.0",
        "dash>=2.16.1",
        "dash-bootstrap-components>=1.5.0",
        "dash-bootstrap-templates==1.1.1",
        "dash_ag_grid",
        "tabulate>=0.9.0",
        "pytest",
        "pytest-sugar",
        "coverage",
        "pytest-cov",
        "flake8",
        "black",
    ],
}

setup(
    name="sageworks",
    description="SageWorks: A Python WorkBench for creating and deploying AWS SageMaker Models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="SuperCowPowers LLC",
    author_email="support@supercowpowers.com",
    url="https://github.com/SuperCowPowers/sageworks",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    package_data={
        "sageworks": [
            "resources/signature_verify_pub.pem",
            "resources/open_source_api.key",
            "core/transforms/features_to_model/light_xgb_model/xgb_model.template",
            "core/transforms/features_to_model/light_xgb_model/requirements.txt",
        ]
    },
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT",
    keywords="SageMaker, Machine Learning, AWS, Python, Utilities",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    setup_requires=["setuptools_scm"],
    entry_points={"console_scripts": "sageworks = sageworks.repl.sageworks_shell:launch_shell"},
)
