# scatt_scripts

Repository for the optimized forward and backward scattering analysis procedures on VESUVIO

Currently in development, daily updates and corrections.

Two samples are provided, D_HMT and starch_80_D, each with its own main script and directories for input and output data.

How to use for a new sample: 

    1. Copy one of the main scripts D_HMT.py or starch_80_RD.py and create a new .py file with the initial conditions of the new sample

    2. Need to create a directory under "experiments" for the new sample, replicating the example directory of DHMT. 
    (This step will be made automatic in the future)

    3. Put forwad and backward workspaces in input_ws under the new sample data directories, they are detected if their names contain "raw", "empty", "backward" and "forward" seperated by "_" and are sorted out accordingly.

    4. After the data directories are sorted out, the main script is ready to be run. The procedures for data reduction can be found on core_functions/procedures.py, so import the preferred functions onto the main script.

    5. God's willing, the main script will run without issues.

    
