# scatt_scripts

Repository for the optimized forward and backward scattering analysis procedures on VESUVIO

Currently in development, daily updates and corrections

Two samples are provided, D_HMT and starch_80_D, each with its own initial conditions scipt, input workspaces and ip file.

How to use: 

    1. Change initial conditions script in the folder of desired sample.

    2. Put forwad and backward workspaces in input_ws, they are detected if their names contain "raw", "empty", "backward" and "forward" seperated by "_" and are sorted out accordingly.

    3. Open run_opt_script.py and at the top need to manually (for now) import objects from the initial conditions file that was changed in point 1. 

    4. Scroll down on run_opt_script.py and select the routine to be run, currently set to runSequenceForKnownRatio() with the IC object from D_HMT.

    
