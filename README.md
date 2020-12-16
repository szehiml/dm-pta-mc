# Dark Matter - Pulsar Timing Array Monte Carlo

## Overview

Monte Carlo (MC) simulations for the Doppler dark matter signals in pulsar timing arrays (PTA). The main module computes the SNR for each of these searches. There is also a supplementary module that computes
the signal shape for a realization.

## Installation

Use a (mini)conda to create an environment with all the necessary dependencies. If you do not have conda installed, install it from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Create the conda environment with:

    conda create --name <env> --file requirements.txt
    conda activate <env>

## Usage

Once in the conda environment, from the main folder run

    mpirun -np 100 python dm-pta-mc.py input.txt

The main module is parallelized over realizations. The argument after '-np' is the number of processors used for the simulation. For optimal performace it should be no more than the number of universes. The last argument passed to the code is the filename of the input parameter file. Some examples are contained
in the examples foler:

- pbh_input.txt: exmple input file for PBH (monochromatic halo mass function, point mass)
- AX_input.txt: example input file for axion minicluster 

Details of each parameter can be found in the example input files.

The output is saved in the /data folder.

<!-- 2. Supplementary module (signal shape) -->

<!-- Go to the 'earth/' or the 'pulsar/' directory. Run the code with -->

<!--     python signal_shape.py input.txt -->

<!-- The output is saved as 'dphi.txt' and 'ht.txt' in the output directory specified in the input file. The first -->
<!-- column is time (in s) and the second column is the time series of the un-subtracted / subtracted dark matter --> 
<!-- signal (in s). Multiply it by the pulsar frequency to obtain the dimensionless signal. -->

Contributors
------------

The MC was authored by Vincent Lee and Tanner Trickle. Contributors include Andrea Mitridate.

Citation
--------

ArXiv link to the paper associated to this code will be posted shortly.
