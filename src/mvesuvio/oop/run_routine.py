
from mvesuvio.oop.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace
from mvesuvio.oop.AnalysisRoutine import AnalysisRoutine


def run_analysis():
     
    ws = loadRawAndEmptyWsFromUserPath()

    cropedWs = cropAndMaskWorkspace()

    AR = AnalysisRoutine() 
    AR.run()

