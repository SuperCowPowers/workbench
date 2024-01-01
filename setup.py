#!/usr/bin/env python
"""Setup.py for SageWorks: Sagemaker Workbench"""

from setuptools import setup, find_packages

# Readme
with open("README.md", "r") as f:
    readme = f.read()

# Requirements
with open("requirements.txt", "r") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="sageworks",
    description="SageWorks: A Python WorkBench for creating and deploying AWS SageMaker Models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="SuperCowPowers LLC",
    author_email="support@supercowpowers.com",
    url="https://github.com/SuperCowPowers/sageworks",
    use_scm_version=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "sageworks": [
            "resources/signature_verify_pub.pem",
            "resources/open_source_api.key",
            "core/transforms/features_to_model/light_model_harness/xgb_model.template",
            "core/transforms/features_to_model/light_model_harness/requirements.txt",
        ]
    },
    install_requires=install_requires,
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
