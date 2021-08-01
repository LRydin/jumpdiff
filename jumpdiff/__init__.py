from .q_ratio import q_ratio
from .kernels import epanechnikov, silvermans_rule
from .moments import moments, corrections
from .jd_process import jd_process
from .parameters import jump_amplitude, jump_rate
from .formulae import m_formula, f_formula, f_formula_solver

name = "jumpdiff"

__version__ = "0.4.2"
__author__ = "Leonardo Rydin Gorjão"
__copyright__ = "Copyright 2019-2021 Leonardo Rydin Gorjão, MIT License"
