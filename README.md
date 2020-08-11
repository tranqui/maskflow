> ### <img src="https://www.emoji.co.uk/files/phantom-open-emojis/symbols-phantom/13025-warning-sign.png" width="35" height="35" /> 11th August 2020: We are continually improving this code and its documentation, so check back later for improvements.

# maskflow
## A python package to evaluate the filtration properties of face masks and coverings.

## Overview

This package accompanies a manuscript available on a preprint server (awaiting submission to an academic journal):

* Robinson *et al* "Efficacy of face coverings in reducing transmission of COVID-19: calculations based on models of droplet capture" [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (2020).

The theory is described in detail there, along with references for further reading.

Bug reports can be posted to the [issues page](https://github.com/tranqui/maskflow/issues) and we will work to fix them as quickly as we can manage. For more general questions/comments, please don't hesitate to contact the authors directly:
- Joshua F. Robinson ([joshua.robinson@bristol.ac.uk](mailto:joshua.robinson@bristol.ac.uk), [GitHub page](https://github.com/tranqui))
- Richard P. Sear ([r.sear@surrey.ac.uk](mailto:r.sear@surrey.ac.uk), [website](https://www.richardsear.me))

## Prerequisites

This package assumes python 3 is your default python executable. If you have other versions of python on your system you may need to run the scripts with the `python3` executable (instead of simply `python`).

This package requires [numpy](https://numpy.org), [scipy](https://scipy.org) and [matplotlib](https://matplotlib.org).

We also use the [YAML file format](https://en.wikipedia.org/wiki/YAML) to store input/outputs in many of the scripts, which we process using the [pyyaml](https://pypi.org/project/PyYAML/) library. Install this for the current user with pip via

    pip install --user pyyaml

A YAML file `example.yml` can then be read through

    import yaml
    with open('example.yml') as f:
        data = yaml.load(f, Loader=yaml.Loader)

The variable `data` will be a `dict` containing all the parameters inside the file.

## Usage

This package determines the efficiency of filters in two different flow fields:
* The Kuwabara flow field, an analytic approximation to the neighbourhood around fibres and are much faster for calculations, and
* Lattice-Boltzmann flow fields, which are essentially exact (subject to the geometrical approximations used in their construction).

These flow fields are described in detail in our manuscript linked to [above](#overview). The next few sections show how to use the default scripts to produce results shown in the manuscript.

### 1) Kuwabara: Single-fibre efficiency

We provide a script [singlefibre.py](single-fibre.py) for assessing single-fibre efficiencies. Inside the source folder navigate to the `maskflow` subfolder, and run e.g.:

    python ./singlefibre.py -df 10 5.325 0.15 -P 1000

Explanation of options in this example:
* `-d` indicates that sizes of particle and fibre are given for diameters (otherwise radius is assumed)
* `-f 10`: this sets the fibre size to ten microns
* `-P 1000`: this will make the program output the final penetration through a mask of 1000 micron (1 mm)

By default the program will perform find the limiting trajectory of particle flow onto the fibre by an optimisation procedure fully incorporating inertia into the dynamics. Some other options for the calculation:
* `-p` would estimate lambda by treating inertia as a perturbation around the background flow field.
* `-a` would estimate lambda using an analytical expression for efficiency due to Stechkina *et al* (1969).

To run this over a range of particle sizes you can directly pass numpy expressions in the particle size field, e.g.:

    python ./singlefibre.py -df 10 "np.linspace(1,5,5)" 0.15

will evaluate the filtration performance of 10 micron diameter fibres with incoming particles of 1, 2, 3, 4 and 5 microns.

Execute this script with the help flag `-h` to see a full list of options.

### 2) Kuwabara: Averaging over a fibre efficiency

To determine the effectiveness of a filter (i.e. the mask or covering) composed of many fibres, we provide the scripts [multifibre.py](maskflow/multifibre.py) and [fabric.py](maskflow/fabric.py). Filters are comprised of polydisperse mixtures of fibres, so we have to take the outputs of [singlefibre.py](maskflow/singlefibre.py) for a range of fibre sizes and interpolate (in some cases extrapolate) over the available data.

To obtain a suitable dataset for interpolation, in Bash you can execute the following:

    mkdir lambda_alpha=0.1_flow=0.075
    for df in $(seq 0.5 0.1 4.9) $(seq 5 0.5 40); do ../../maskflow/singlefibre.py -df $df "np.arange(0.1,40,0.1)" 0.1 -F 0.075 -YR --face > lambda_alpha=0.1_flow=0.075/df=$df.yml; done

> <img src="https://www.emoji.co.uk/files/phantom-open-emojis/symbols-phantom/13025-warning-sign.png" width="15" height="15" /> **Warning: the above command can take a reasonably long time to execute; even with a high-performance machine it takes several days to complete. Running the procedures in parallel would greatly speed up this process (and is recommended if possible).**

In addition to arguments seen in the previous section, this example introduces the following arguments:
* `-F` indicates the next argument is the flow speed in m/s (7.5cm/s in this case).
* `--face` states the flow speed is interpretted as the flow speed before entering the filter (also known as the "face speed"). The flow speed inside the filter will be this number divided by `(1-a)` where `a` is the volume fraction (0.1 in the above example).
* `-Y` forces the plain text output into a YAML-compliant format, which makes it easier to parse the output data via a program (cf. [prerequisites](#prerequisites) for an example of this using [pyyaml](https://pypi.org/project/PyYAML/)).

After running the above example the folder `lambda_alpha=0.1_flow=0.075` will have been created containing the data files needed to average using the script [multifibre.py](maskflow/multifibre.py). This script assumes that the fibre diameters are distributed [log-normally](https://en.wikipedia.org/wiki/Log-normal_distribution). To average over such a distribution with modal value 14.5 microns and standard deviation 0.34 (the parameters for cotton, cf. the manuscript) run:

    python multifibre.py 14.5 0.34 -p lambda_alpha=0.1_flow=0.075/*.yml

which will plot the sampled values of lambda and the average values. This plot also indicates the region where we have to use an extrapolation for lambda (where the particle size approaches the fibre size) because the Kuwabara flow field is pathological in this limit; this does not affect the end result, because the penetration is approximately zero in this limit. We strongly discourage skipping this step, because it illustrates the inner workings of the averaging procedure.

To obtain usable data for the next step, run

    python multifibre.py 14.5 0.34 -Y lambda_alpha=0.1_flow=0.075/*.yml > alpha=0.1_flow=0.075_df=14.5_s=0.34.yml

Run `python multifibre.py -h` for a complete list of options.

### 3) Kuwabara: Total mask efficiency

For the final step we take the results of previous steps and determine the fabric filter efficiency versus particle size. The following script will read in the results of the previous example, and plot the collection efficiency for a 1mm thick cotton fabric:

    import numpy as np
    import matplotlib.pyplot as plt
    from maskflow.fabric import Fabric
    
    dp = np.logspace(-9, -4, 1001) # particle diameters in metres
    L = 1e-3 # filter thickness in metres
    fabric = Fabric('alpha=0.1_flow=0.075_df=14.5_s=0.34.yml') # output of previous step

    plt.semilogx(1e6*dp, 1-fabric.penetration(dp, L))
    plt.xlabel('particle diameter (um)')
    plt.ylabel('collection efficiency')
    plt.ylim([0, 1])
    plt.show()

### Lattice-Boltzmann simulations

To calculate a lambda:

1. First run LB code to obtain a flow field.
Edit the params.yaml to set all the parameters needed for the flow field - which is all parameters except the particle diameter. To calculate the flow field for a lambda calculation set single_fibre to True. From the source directory navigate to the `maskflow` subfolder then execute:

        python ./latticeboltzmann.py

    params.yaml in same directory. You should check that towards the end of the run, the flow field is hardly changing. It should then be near steady state. The LB code will write out a .npz file

2. Second obtain lambda for given particle diameter, in the LB flow field by executing:

        python ./lbparticletraj.py

    for the required particle diameter, with the .npz in the same directory. This will calculate the lambda.

At the end the code writes to lambda.txt, one line with, in order:

* Stokes number
* ratio particle diameter/fibre diameter
* fibre diameter in micrometres
* particle diameter in micrometres
* lambda in micrometres
* LB lattice constant in micrometres
* alpha (area fraction)


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

## Using the package as a library

Instead of running the pre-built scripts, you can install the package and import the modules for use in your own code. To install the package, navigate to the source directory and execute:

    pip install --user .

Then you should be able to import the various modules in your own code, e.g.

    from maskflow import kuwabara
    flow_field = kuwabara.KuwabaraFlowField(alpha=0.1)

will create a Kuwbara flow field with an occupied volume fraction of 10%. See our manuscript linked to [above](#overview) for description of the theory, and the code itself for documentation of functionality.
