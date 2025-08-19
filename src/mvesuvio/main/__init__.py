"""Package defining entry points."""

import argparse
from os import path
from mvesuvio.util import handle_config


def main(manual_args=None):
    args = __setup_and_parse_args() if not manual_args else manual_args
    if args.command == "config":
        __setup_config(args)

    if args.command == "run":
        if not handle_config.config_set():
            __setup_config(None)
        __run_analysis(args)


def __setup_and_parse_args():
    parser = __set_up_parser()
    args = parser.parse_args()
    return args


def __set_up_parser():
    parser = argparse.ArgumentParser(description="Package to analyse Vesuvio instrument data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("config", help="set mvesuvio configuration")
    config_parser.add_argument("--set-inputs", "-i", help="set the inputs python file", default="", type=str)
    config_parser.add_argument(
        "--set-ipfolder",
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
    return parser


def __setup_config(args):
    __set_logging_properties()

    config_dir = handle_config.VESUVIO_CONFIG_PATH
    handle_config.setup_config_dir(config_dir)
    ipfolder_dir = handle_config.VESUVIO_IPFOLDER_PATH
    inputs = handle_config.VESUVIO_INPUTS_PATH

    if handle_config.config_set():
        inputs = handle_config.read_config_var("caching.inputs") if not args or not args.set_inputs else args.set_inputs
        ipfolder_dir = handle_config.read_config_var("caching.ipfolder") if not args or not args.set_ipfolder else args.set_ipfolder
    else:
        inputs = inputs if not args or not args.set_inputs else args.set_inputs
        ipfolder_dir = ipfolder_dir if not args or not args.set_ipfolder else args.set_ipfolder

        handle_config.setup_default_ipfile_dir()
        handle_config.setup_default_inputs()

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


def __run_analysis(args):
    from mvesuvio.main.run_routine import Runner

    if not args:
        Runner().run()
        return
    Runner(override_back_workspace=args.back_workspace, override_front_workspace=args.front_workspace).run()


if __name__ == "__main__":
    main()
