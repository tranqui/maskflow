# maskflow
## A python package to evaluate filtration properties of face masks and coverings.

## Overview

> ## ![](http://www.emoji.co.uk/files/phantom-open-emojis/symbols-phantom/13025-warning-sign.png) 11th August 2020: We are continually improving this code and its documentation, so check back later for improvements

This package accompanies a manuscript available on a preprint server (awaiting submission to an academic journal):

* Robinson *et al* "Efficacy of face coverings in reducing transmission of COVID-19: calculations based on models of droplet capture" [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (2020).

The theory is described in detail there, along with references for further reading.

If you have any questions/issues, please don't hesitate to contact the authors directly:
- Joshua F. Robinson ([joshua.robinson@bristol.ac.uk](mailto:joshua.robinson@bristol.ac.uk), [GitHub page](https://github.com/tranqui))
- Richard P. Sear ([r.sear@surrey.ac.uk](mailto:r.sear@surrey.ac.uk), [website](https://www.richardsear.me))


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

To calculate a lambda:

1. First run LB code to obtain a flow field.
Edit the params.yaml to set all the parameters needed for the flow field - which is all parameters except the particle diameter. To calculate the flow field for a lambda calculation set single_fibre to True. Then run LB_maskAug2020.py with:

        python ./LB_maskAug2020.py

params.yaml in same directory. Here I assume you have python3 is default, if not try python3. You should check that towards the end of the run, the flow field is hardly changing. It should then be near steady state. The LB code will write out a .npz file

2. Second run traj_calc_maskAug2020.py to obtain lambda for given particle diameter, in the LB flow field:

        python ./traj_calc_maskAug2020.py

for the required particle diameter, with the .npz in the same directory. This will calculate the lambda.

At the end the code writes to lambda.txt, one line with, in order:

Stokes number
ratio particle diameter/fibre diameter
fibre diameter in micrometres
particle diameter in micrometres
lambda in micrometres
LB lattice constant in micrometres
alpha (area fraction)


To calculate a penetration:

1. As above except set single_fibre to False in params.yaml. LB code will then generate a small disorderd lattice of discs and calculate the flow field around them. Note that as lattice has random disorder, running the program twice will give different lattices (unlike for the single fibre case).
2. As above, run the trajectory code for the required particle diameter, with the .npz in the same directory. This will output the penetration. It just starts off particles on an equally spaced grid and the penetration is then estimated as the fraction of these particles that get through.

At the end the code writes to filter_eff.txt, one line with, in order:

* Stokes number
* ratio particle diameter/fibre diameter
* fibre diameter in micrometres
* particle diameter in micrometres
* penetration (fraction of trajectories that penetrated)
* LB lattice constant in micrometres
* alpha (area fraction)
* filter thickness in LB lattice units