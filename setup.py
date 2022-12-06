#!/usr/bin/env python
"""Setup.py for SageWorks: Sagemaker Workbench"""

import os
import glob

from setuptools import setup, find_packages

readme = open('README.md').read()


# Data and Example Files
def get_files(dir_name):
    """Simple directory walker"""
    return [(os.path.join('.', d), [os.path.join(d, f) for f in files]) for d, _, files in os.walk(dir_name)]


setup(
    name='sageworks',
    # use_scm_version=True,
    version='0.1.0',
    description='SageWorks: An easy to use WorkBench for using and deploying SageMaker Models',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='SuperCowPowers LLC',
    author_email='support@supercowpowers.com',
    url='https://github.com/SuperCowPowers/sageworks',
    packages=find_packages('src'),
    package_dir={"": "src"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")],
    include_package_data=True,
    data_files=get_files('data') + get_files('examples'),
    install_requires=[
        'sagemaker',
        'pandas',
        'scikit-learn'
    ],
    license='Apache License 2.0',
    keywords='SageMaker, Machine Learning, AWS, Python, Utilities',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    setup_requires=["setuptools_scm", "setuptools"]
)
