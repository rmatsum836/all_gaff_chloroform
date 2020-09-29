# all_gaff_chloroform

This is a signac projects used to simulate bulk solutions of acetonitrile, chloroform, and LiTFSI.

## Simulation Details
- GAFF Force field for acetonitrile, chloroform, and TFSI- anion
- Amber Force field for Li+ ion
- Acetonitrile, chloroform, and TFSI- partial charges calculated through RESP using Gaussian09 and the R.E.D Server
- Li+ and TFSI- partial charges scaled by 0.8
- Simulations run with GROMACS 2018.5

## Usage
The signac project can be initialized by running `python src/init.py` on the command line.  This command initializes the project and creates all of the necessary signac `jobs`.  Various commands can be executed by running `python src/project.py [command]`.

All data managed by signac can be impored to a csv file by running: `python dataframe.py`.  All plotting functions are contained within `plot_pandas.py`.

### Requirements
The list of packages and dependencies can be viewed and installed via the `environment.yml` file.

### Use of the R.E.D Server
The R.E.D Server was used to derive the partial charges for the acetonitrile, chloroform, and TFSI-.  To reproduce this work, start by going to the R.E.D Server [here](https://upjv.q4md-forcefieldtools.org/REDServer-Development/) and clicking the `submit` button at the top right.  Once the pop-up appears, click on the `private account` link at the bottom to submit a job to the server using your account (Create one if you haven't done so already).  The program `Gaussian09E.01` was used for this work.  The tarballs for the different molecules are contained withing the `red_server` directory as well as the uncompressed folders.
