import yaml


def load_yaml_config(args, default_args):

    if args.config is None:
        return args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.items():

        if not hasattr(args, key):
            continue

        # Only replace if the user didn't override the CLI value
        if getattr(args, key) == getattr(default_args, key):
            setattr(args, key, value)

    return args