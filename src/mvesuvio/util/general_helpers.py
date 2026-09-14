import re

from mantid.kernel import logger

from mvesuvio.util.files_manager import FilesManager


def make_summarised_log_file() -> None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    try:
        with open(FilesManager.get_mantid_log_file(), "r") as infile, open(FilesManager.get_summarised_log_file(), "w") as outfile:
            for line in infile:
                if "VesuvioAnalysisRoutine" in line:
                    outfile.write(line)

                if "Notice Python" in line:  # For Fitting notices
                    outfile.write(line)

                if not pattern.match(line):
                    outfile.write(line)
    except OSError:
        logger.error("Mantid log file not available. Unable to produce a summarized log file for this routine.")
    return
