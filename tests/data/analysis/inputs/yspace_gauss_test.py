from tests.data.analysis.inputs.analysis_test import (
    BackwardInitialConditions,
    ForwardInitialConditions,
    YSpaceFitInitialConditions,
)
ForwardInitialConditions.noOfMSIterations = 1
ForwardInitialConditions.firstSpec = 164
ForwardInitialConditions.lastSpec = 175
ForwardInitialConditions.fit_in_y_space = True 
BackwardInitialConditions.fit_in_y_space = False
ForwardInitialConditions.run_this_scattering_type = True 
BackwardInitialConditions.run_this_scattering_type = False
YSpaceFitInitialConditions.fitModel = "SINGLE_GAUSSIAN"
