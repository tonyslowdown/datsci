"""Exploratory Data Analysis
"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.08.03
# Last Modified   : 2016.08.05
#
# License         : MIT

import sys
from PIL import Image


def extract_ch(imgfile, ch='r'):
    """Extract a channel in an image
    """
    im = Image.open(imgfile).convert('RGB')

    # Determine which of the rgb values to take
    if ch in {'r', 'R'}:
        rgb_idx = 0
    elif ch in {'g', 'G'}:
        rgb_idx = 1
    elif ch in {'b', 'B'}:
        rgb_idx = 2
    elif ch in {'a', 'A'}:
        rgb_idx = 3

    return im.split()[rgb_idx]


def print_ch(imgfile, ch='r', writeable=sys.stdout):
    """Print the channel values of a given image file
    """
    im_ch = extract_ch(imgfile, ch=ch)
    w, h = im_ch.size
    for i in range(h):
        for j in range(w):
            val = im_ch.getpixel((j, i))
            sys.stdout.write('{num:03d} '.format(num=val))
        sys.stdout.write("\n")
