#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
required_modules = ["numpy", "scipy", "natsort"]
for module in required_modules:
    try: exec("import %s" % module)
    except: requirements += [module]

setuptools.setup(
    name="maskflow",
    version="v1.0.0",
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
