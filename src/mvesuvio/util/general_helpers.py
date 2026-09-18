import re

from mantid.kernel import logger

from mvesuvio.util.files_manager import FilesManager


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
