"""Package defining entry points."""

import argparse
from os import path
from pathlib import Path
from mvesuvio.util import handle_config


def main(manual_args=None):
    args = _setup_and_parse_args() if not manual_args else manual_args
    if args.command == "config":
        _setup_config(args)

    if args.command == "run":
        if not handle_config.config_set():
            _setup_config(None)
        _run_analysis(args)

    if args.command == "bootstrap":
        if not handle_config.config_set():
            _setup_config(None)
        _run_bootstrap(args)


def _setup_and_parse_args():
    parser = _set_up_parser()
    args = parser.parse_args()
    return args


def _set_up_parser():
    parser = argparse.ArgumentParser(description="Package to analyse Vesuvio instrument data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("config", help="set mvesuvio configuration")
    config_parser.add_argument("--analysis-inputs", "-i", help="set the inputs python file", default="", type=str)
    config_parser.add_argument(
        "--ip-folder",
        "-p",
        help="set the intrument parameters directory",
        default="",
        type=str,
    )

    run_parser = subparsers.add_parser("run", help="run mvesuvio analysis")
    run_parser.add_argument(
        "--back-workspace",
        "-b",
        help="input workspace for vesuvio backward analysis, bypasses loading (and subtracting) raw and empty.",
        default="",
        type=str,
    )
    run_parser.add_argument(
        "--front-workspace",
        "-f",
        help="input workspace for vesuvio forward analysis, bypasses loading (and subtracting) raw and empty.",
        default="",
        type=str,
    )
    run_parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Flag to set output files to minimum.",
        default=False,
    )
    run_parser.add_argument(
        "--outputs-dir",
        "-o",
        help="Directory for populating with output files.",
        default="",
        type=str,
    )
    boot_parser = subparsers.add_parser("bootstrap", help="Run bootstrap of vesuvio analysis (without y-space fitting)")
    boot_parser.add_argument(
        "--inputs-dir",
        "-d",
        help="Directory containing input bootstrap replicas. Replicas should be inside sparate backaward and forward subdirectories.",
        default="",
        type=str,
    )
    return parser


def _setup_config(args):
    __set_logging_properties()

    handle_config.refresh_config_dir_and_contents()

    if not handle_config.config_set():
        handle_config.set_default_config_vars()

    inputs = handle_config.read_config_var("caching.inputs")
    ipfolder_dir = handle_config.read_config_var("caching.ipfolder")

    if args and args.set_inputs:
        inputs = str(Path(args.set_inputs).absolute())
    if args and args.set_ipfolder:
        ipfolder_dir = str(Path(args.set_ipfolder).absolute())

    handle_config.set_config_vars(
        {
            "caching.inputs": inputs,
            "caching.ipfolder": ipfolder_dir,
        }
    )
    handle_config.check_dir_exists("IP folder", ipfolder_dir)


def __set_logging_properties():
    from mantid.kernel import ConfigService

    ConfigService.setString("logging.loggers.root.channel.class", "SplitterChannel")
    ConfigService.setString("logging.loggers.root.channel.channel1", "consoleChannel")
    ConfigService.setString("logging.loggers.root.channel.channel2", "fileChannel")
    ConfigService.setString("logging.channels.consoleChannel.class", "ConsoleChannel")
    ConfigService.setString("logging.channels.fileChannel.class", "FileChannel")
    ConfigService.setString("logging.channels.fileChannel.path", "mantid.log")
    ConfigService.setString("logging.channels.fileChannel.formatter.class", "PatternFormatter")
    ConfigService.setString("logging.channels.fileChannel.formatter.pattern", "%Y-%m-%d %H:%M:%S,%i [%I] %p %s - %t")
    # Set properties on Mantid.user.properties not working due to Mantid bug
    # Need to set properties on file in Mantid installation
    mantid_properties_file = path.join(ConfigService.getPropertiesDir(), "Mantid.properties")
    ConfigService.saveConfig(mantid_properties_file)
    return


def _run_analysis(args):
    from mvesuvio.main.run_routine import Runner

    if not args:
        Runner().run()
        return
    Runner(
        override_back_workspace=args.back_workspace,
        override_front_workspace=args.front_workspace,
        minimal_output=args.minimal_output,
        output_directory=args.outputs_dir,
    ).run()


def _run_bootstrap(args):
    from mvesuvio.main.run_routine import Runner

    if not args:
        return
    Runner(bootstrap_inputs_directory=args.inputs_dir, minimal_output=True).run_bootstrap()


if __name__ == "__main__":
    main()
