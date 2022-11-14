import sys
import numpy
import scipy
import sympy

def versioning_libraries():
    print('python:', sys.version)
    print('numpy:', numpy.__version__)
    print('scipy:', scipy.__version__)
    print('sympy:', sympy.__version__)
