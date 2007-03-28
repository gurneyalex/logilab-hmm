

import sys
from os.path import dirname, join, abspath

module = sys.modules[__name__].__file__
topdir = abspath(join( dirname(module), "..", "..", ".." ))

sys.path.insert(0,topdir)
