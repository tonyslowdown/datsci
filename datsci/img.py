"""Exploratory Data Analysis
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.08.03
# Last Modified   : 2016.08.03
#
# License         : MIT

import sys
from PIL import Image


def print_ch(imgfile, ch='r', writeable=sys.stdout):
    """Print the channel values of a given image file
    """
    im = Image.open(imgfile).convert('RGB')
    w, h = im.size

    # Determine which of the rgb values to take
    if ch in {'r', 'R'}:
        rgb_idx = 0
    elif ch in {'g', 'G'}:
        rgb_idx = 1
    elif ch in {'b', 'B'}:
        rgb_idx = 2

    for i in range(h):
        for j in range(w):
            val = im.getpixel((j, i))[rgb_idx]
            writeable.write('{num:03d} '.format(num=val))
        writeable.write('\n')
