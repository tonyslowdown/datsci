"""Command line tool
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.08.03
# Last Modified   : 2016.08.03
#
# License         : MIT

import argparse
import inspect
from datsci import img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Datsci command-line tool')
    subparsers = parser.add_subparsers(
        title='subcommands',
        description='valid subcommands')

    # Parser corresponding to `img` module
    img_parser = subparsers.add_parser(
        'image',
        help='Image module')
    img_cmds = img_parser.add_subparsers(
        title='image subcommands',
        description='valid image subcommands')

    # Parser corresponding to `img.print_channel`
    img_printch_parser = img_cmds.add_parser(
        'print_channel', aliases=['pc'],
        help='Print the color values of each pixel for a given channel, \
        for a given image')
    img_printch_parser.add_argument(
        'imgfile', type=str,
        help='Image file path')
    img_printch_parser.add_argument(
        '-ch', '--channel', dest='ch', choices=['r', 'g', 'b'], default='r')
    img_printch_parser.set_defaults(func=img.print_ch)

    # Parse arguments
    args = parser.parse_args()
    arg_spec = inspect.getargspec(args.func)
    kwargs = {k: getattr(args, k) for k in arg_spec.args if hasattr(args, k)}
    args.func(**kwargs)
