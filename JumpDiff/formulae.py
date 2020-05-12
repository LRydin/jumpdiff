

from sympy import bell, symbols, factorial, simplify

def M_formula(power, tau = True):
    from sympy import bell, symbols, factorial, simplify

    init_sym = symbols('D:'+str(int(power+1)))[1:]
    sym = ()
    for i in range(1,power + 1):
        sym += (factorial(i)*init_sym[i-1],)

    if tau == True:
        t = symbols('tau')

        term = t*bell(power, 1, sym) + t**2*bell(power, 2, sym)

    else:
        term = bell(power, 1, sym) + bell(power, 2, sym)

    return term


M_formula(5)


sym = symbols('M:'+str(int(8+1)))[1:]
sym[0]
def F_formula_gen(power):
    from sympy import bell, symbols, factorial, simplify

    init_sym = symbols('D:'+str(int(power+1)))[1:]
    sym = ()
    for i in range(1,power + 1):
        sym += (factorial(i)*init_sym[i-1],)

    term = (symbols('M'+str(int(power))) - bell(power, 2, sym))/factorial(power)

    return term


def F_formula_solver(power):
    from sympy import bell, symbols, factorial, simplify

    power = power + 1
    terms_to_sub = []
    for i in range(1, power):
        terms_to_sub += [F_formula_gen(i)]
        for j in range(i):
            terms_to_sub[i-1] = terms_to_sub[i-1].subs('D'+str(j+1),terms_to_sub[j])

    # for i in range
    # term = (symbols('M'+str(int(power))) + bell(power, 2, sym))/factorial(power)

    return (terms_to_sub[power-2]*factorial(power-1)).expand(basic = True)

F_formula_solver(5)


from sympy import bell, symbols, factorial, simplify
power = 5

# %%
terms_to_sub = []

for i in range(1, power-1):
    terms_to_sub += [F_formula_gen(i)]

# %%

sym[0]
F_formula_gen(2).subs('D1', sym[0]+sym[1])


power = 7
terms_to_sub = []
for i in range(1, power):
    terms_to_sub += [F_formula_gen(i)]
    for j in range(i):
        terms_to_sub[i-1] = terms_to_sub[i-1].subs('D'+str(j+1),terms_to_sub[j])

terms_to_sub
simplify(terms_to_sub[5]*factorial(6)).expand(basic = True)
