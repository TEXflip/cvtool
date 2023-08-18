import argparse
from cvtool.__main__ import logger

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class ActionRectify(argparse.Action):

    opt_params = {
        "rect_points" : ["points", "p"],
    }

    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 required=False,
                 help=None):
        
        super(ActionRectify, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs='*',
            default=default,
            required=required,
            help=help)
        
    def __call__(self, parser, namespace, values, option_string=None):
        default = {"original_size": True}
        for value in values:
            if isfloat(value):
                if "rect_points" in default:
                    default["rect_points"].append(float(value))
                else:
                    default["original_size"] = float(value)
            elif value in self.opt_params["rect_points"]:
                default["rect_points"] = []
            else:
                logger.warning(f"Invalid parameter: {value}")

        if "rect_points" in default:
            if len(default["rect_points"]) != 8:
                logger.warning(f"Invalid points parameter: {default['rect_points']}")
                del default["rect_points"]
            else:
                p = default["rect_points"]
                default["rect_points"] = [[p[2 * i], p[2 * i + 1]] for i in range(4)]

        setattr(namespace, self.dest, default)