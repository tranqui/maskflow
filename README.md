# maskflow
## Python package to evaluate filtration properties of face masks and coverings.

## Installation

Inside the source directory execute:

    pip install --user .

## Usage

### Single-fibre efficiency

We provide a script `single-fibre.py` for assessing single-fibre efficiencies. Inside the source folder navigate to the `maskflow` subfolder, and run e.g.:

    ./single-fibre.py -df 10 5.325 0.15 -P 1000

Explanation of options in this example:
- `-d` indicates that sizes of particle and fibre are given for diameters (otherwise radius is assumed)
- `-f 10`: this sets the fibre size to ten microns
- `-P 1000`: this will make the program output the final penetration through a mask of 1000 micron (1 mm)

By default the program will perform find the limiting trajectory of particle flow onto the fibre by an optimisation procedure fully incorporating inertia into the dynamics. Alternatively, passing the option `-p` will force the script to treat inertia as a perturbation around the background flow field. Finally, the option `-a` will instead use an analytical expression for efficiency due to Stechkina (1969).

To run this over a range of particle sizes you can directly pass numpy expressions in the particle size field, i.e.:

    ./single-fibre.py -df 10 "np.linspace(1,5,5)" 0.15

Execute this script with the help flag `-h` to see a full list of options.

### Total mask efficiency

TBD.

### Lattice-Boltzmann simulations

TBD
