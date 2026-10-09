import re
from typing import Any

from mantid.kernel import logger

from mvesuvio.util.files_manager import FilesManager


def _infer_scattering_type(inputs_class: Any) -> str:
    """Infer scattering type from user inputs without importing reduction helpers."""
    name = str(getattr(inputs_class, "name", "")).lower()
    class_name = getattr(inputs_class, "__name__", "").lower()
    mode = str(getattr(inputs_class, "mode", "")).lower().replace(" ", "")

    if name.startswith("back") or "backward" in class_name or mode == "doubledifference":
        return "backward"

    if name.startswith("front") or "forward" in class_name or mode == "singledifference":
        return "forward"

    raise ValueError(f"Could not infer scattering type for input class: {inputs_class}")


def pass_data_into_ws(dataX, dataY, dataE, ws):
    "Modifies ws data to input data"
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def extractWS(ws):
    """Directly extracts data from a workspace into arrays."""
    return ws.extractX(), ws.extractY(), ws.extractE()


def print_table_workspace(table, precision=3):
    table_dict = table.toDict()
    # Convert floats into strings
    for key, values in table_dict.items():
        new_column = [int(item) if (isinstance(item, float) and item.is_integer()) else item for item in values]
        table_dict[key] = [f"{item:.{precision}f}" if isinstance(item, float) else str(item) for item in new_column]

    max_spacing = [max([len(item) for item in values] + [len(key)]) for key, values in table_dict.items()]
    header = "|" + "|".join(f"{item}{' ' * (spacing - len(item))}" for item, spacing in zip(table_dict.keys(), max_spacing)) + "|"
    logger.notice(f"Table {table.name()}:")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    logger.notice(header)
    for i in range(table.rowCount()):
        table_row = "|".join(
            f"{values[i]}{' ' * (spacing - len(str(values[i])))}" for values, spacing in zip(table_dict.values(), max_spacing)
        )
        logger.notice("|" + table_row + "|")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    return


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


def inject_bootstrap_workspace(analysis_inputs: Any, injected_globals: dict | None = None) -> None:
    """Inject bootstrap workspace override values into analysis input classes."""

    if injected_globals is None:
        injected_globals = {}

    scattering_type = _infer_scattering_type(analysis_inputs)
    if scattering_type == "backward":
        override = injected_globals.get("BACK_OVERWRITE_ANALYSIS_INPUT_WORKSPACE", "")
    else:
        override = injected_globals.get("FRONT_OVERWRITE_ANALYSIS_INPUT_WORKSPACE", "")

    if override:
        analysis_inputs.overwrite_analysis_input_workspace = str(override)
