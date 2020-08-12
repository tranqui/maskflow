#!/usr/bin/env python3

import setuptools
import subprocess

version = subprocess.check_output(["git", "describe", "--tags"]).decode('ascii').strip()

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
required_modules = ["numpy", "scipy", "natsort"]
for module in required_modules:
    try: exec("import %s" % module)
    except: requirements += [module]

setuptools.setup(
    name="maskflow",
    version=version,
    license='GNU General Public License v3.0',

    author="Joshua F. Robinson",
    author_email='joshua.robinson@bristol.ac.uk',

    url='https://github.com/tranqui/maskflow.git',
    description="Code to evaluate filtration properties of face masks.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    python_requires='>=3',
    packages=setuptools.find_packages(),
    package_data = {'': ['*.csv']},
    include_package_data = True,
    install_requires=requirements,

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
