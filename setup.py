#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "scipy"]

setuptools.setup(
    name="maskflow",
    version="0.1",
    author="joshuarrr",
    install_requires=requirements,
    author_email="joshuarrr@protonmail.com",
    description="Code to evaluate filtration properties of face masks.",
    long_description=long_description,
    license='GPLv3',
    long_description_content_type="text/markdown",
    python_requires='>=3',
    packages=setuptools.find_packages(),
    package_data = {'': ['*.csv']},
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
