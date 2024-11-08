from tests.data.analysis.inputs.analysis_test import (
    LoadVesuvioBackParameters,
    LoadVesuvioFrontParameters,
    GeneralInitialConditions,
    BackwardInitialConditions,
    ForwardInitialConditions,
    YSpaceFitInitialConditions,
    UserScriptControls
)

ForwardInitialConditions.noOfMSIterations = 1
ForwardInitialConditions.firstSpec = 164
ForwardInitialConditions.lastSpec = 175
YSpaceFitInitialConditions.fitModel = "SINGLE_GAUSSIAN"
UserScriptControls.runRoutine = True
UserScriptControls.procedure = "FORWARD" 
UserScriptControls.fitInYSpace = "FORWARD"
