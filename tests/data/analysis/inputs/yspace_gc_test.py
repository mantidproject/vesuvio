from tests.data.analysis.inputs.analysis_test import (
    BackwardAnalysisInputs,
    ForwardAnalysisInputs,
    YSpaceFitInputs,
)

ForwardAnalysisInputs.noOfMSIterations = 1
ForwardAnalysisInputs.firstSpec = 164
ForwardAnalysisInputs.lastSpec = 175
ForwardAnalysisInputs.maskedSpecAllNo = [173, 174]
ForwardAnalysisInputs.fit_in_y_space = True 
BackwardAnalysisInputs.fit_in_y_space = False
ForwardAnalysisInputs.run_this_scattering_type = True 
BackwardAnalysisInputs.run_this_scattering_type = False
YSpaceFitInputs.fitModel = "GC_C4_C6"
YSpaceFitInputs.symmetrisationFlag = False 


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    import mvesuvio
    from pathlib import Path
    mvesuvio.set_config(inputs_file=Path(__file__))
    mvesuvio.run()
