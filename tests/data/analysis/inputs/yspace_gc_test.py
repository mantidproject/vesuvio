from tests.data.analysis.inputs.analysis_test import (
    BackwardAnalysisInputs,
    ForwardAnalysisInputs,
    YSpaceFitInputs,
)

ForwardAnalysisInputs.fit_in_y_space = True 
BackwardAnalysisInputs.fit_in_y_space = False
ForwardAnalysisInputs.run_this_scattering_type = True 
BackwardAnalysisInputs.run_this_scattering_type = False
YSpaceFitInputs.fitting_model = "gcc4c6"
YSpaceFitInputs.do_symmetrisation = False 


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    import mvesuvio
    from pathlib import Path
    mvesuvio.set_config(inputs_file=Path(__file__))
    mvesuvio.run()
