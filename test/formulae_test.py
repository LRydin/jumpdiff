import numpy as np
from jumpdiff.formulae import m_formula, f_formula, f_formula_solver

def test_fomulae():
    for power in [1,2,3,4,5,6,7,8,9]:

        m_formula(power = power)

        f_formula(power = power)

        f_formula_solver(power = power)
