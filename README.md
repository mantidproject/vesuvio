# scatt_scripts

Repository for the optimized forward and backward scattering analysis procedures on VESUVIO

Currently in development, daily updates and corrections.

Two example scripts are provided, D_HMT and starch_80_D, each detailing the initial conditions

How to use for a new sample: 

    1. Copy one of the main scripts D_HMT.py or starch_80_RD.py and create a new .py file with the name of the desired sample, in the sample directory as D_HMT.py or starch_80_RD.py. 

    2. Fill in the new script with the desired initial conditions.

    3. Run the script. The script will create a new directory for the sample under experiments/, and will try to use LoadVesuvio to store the workspaces locally for future runs. If LoadVesuvio fails, the user needs to copy the worksapces as .nxs files onto experiments/sample/input_ws/, using the same format as the two example samples provided.

    4. After the workspaces are stored locally, any further data reduction will run using the local workspaces. 

  

    
