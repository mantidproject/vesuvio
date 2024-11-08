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
YSpaceFitInitialConditions.fitModel = "GC_C4_C6"
YSpaceFitInitialConditions.symmetrisationFlag = False 
UserScriptControls.runRoutine = True
UserScriptControls.procedure = "FORWARD" 
UserScriptControls.fitInYSpace = "FORWARD"
