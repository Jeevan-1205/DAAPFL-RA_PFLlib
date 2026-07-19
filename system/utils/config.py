import sys
import yaml


def _explicit_cli_dests(parser, cli_args=None):
    if cli_args is None:
        cli_args = sys.argv[1:]

    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit = set()
    for token in cli_args:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])

    return explicit


def load_yaml_config(args, default_args, parser=None):

    if args.config is None:
        return args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    explicit_dests = _explicit_cli_dests(parser) if parser is not None else None

    for key, value in cfg.items():

        if not hasattr(args, key):
            continue

        if explicit_dests is not None:
            if key not in explicit_dests:
                setattr(args, key, value)
            continue

        # Only replace if the user didn't override the CLI value
        if getattr(args, key) == getattr(default_args, key):
            setattr(args, key, value)

    return args
