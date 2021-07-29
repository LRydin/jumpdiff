import numpy as np
import matplotlib.pyplot as plt

# This is a short python script to play around with JumpDiff
import jumpdiff as jd

# %% Let's first integrate a jump-diffusion process
# Define:
# integration time and time sampling
t_final = 10000
delta_t = 0.001

# A drift function
def a(x):
    return -0.5*x

# and a (constant) diffusion term
def b(x):
    return 0.75

# Now define a jump amplitude and rate
xi = 2.5
lamb = 1.75

# and let jdprocess integrate the stochastic path
X = jd.jdprocess(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb)


# %% Plot the trajectory of the jump-diffusion process
fig, ax = plt.subplots(1,1,figsize=(6,3))

ax.plot(np.linspace(0,t_final,int(t_final/delta_t)), X, color = 'black')
ax.set_xlabel('t', fontsize=16)
ax.set_ylabel('x', fontsize=16)


# %% To retrieve the moments, use:
edges, moments = jd.moments(timeseries = X, bw = 0.35)
# and don't forget that the Kramers─Moyal coefficient need `moments/delta_t`


# %% Let us plot the first Kramers─Moyal coefficient 'moments[1,...]/delta_t'
fig, ax = plt.subplots(1,1,figsize=(6,3))

ax.plot(edges, moments[1,...]/delta_t, color = 'black', label = '1st Kramers─Moyal coefficient')
ax.plot(edges, a(edges), '--', color = 'black', label = 'Theoretical curve a = -0.5*x')
ax.set_xlim([-5,5]); ax.set_ylim([-5,5])

ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('$D_1$(x)', fontsize=16)
ax.legend(fontsize=13)


# %% The second Kramers─Moyal coefficient 'moments[2,...]/delta_t'
fig, ax = plt.subplots(1,1,figsize=(6,3))

ax.plot(edges, moments[2,...]/delta_t, color = 'black', label = '2nd Kramers─Moyal coefficient')
ax.plot(edges, (b(0)**2 + xi*lamb)*np.ones_like(edges), '--', color = 'black', label = 'Theoretical curve $b^2+λξ$')
ax.set_xlim([-5,5]); ax.set_ylim([3,8])

ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('$D_2$(x)', fontsize=16)
ax.legend(fontsize=13)


# %% And the fourth Kramers─Moyal coefficient 'moments[4,...]/delta_t'
fig, ax = plt.subplots(1,1,figsize=(6,3))

ax.plot(edges, moments[4,...]/delta_t, color = 'black', label = '4th Kramers─Moyal coefficient')
ax.plot(edges, (3*(xi**2)*lamb)*np.ones_like(edges), '--', color = 'black', label = 'Theoretical curve $3λξ^2$')
ax.set_xlim([-5,5]); ax.set_ylim([0,60])

ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('$D_4$(x)', fontsize=16)
ax.legend(fontsize=13)


# %% Finally, we can use simply the 'jump_amplitude' and 'jump_rate' functions
# to recover the ξ and λ parameters

xi_est = jd.jump_amplitude(moments = moments)

lamb_est = jd.jump_rate(moments = moments, xi_est = xi)

# Don't forget that the jump rate λ needs to be divide by 'delta_t' to yield a
# comparible result.


# %% If one wishes to check the formulae behind the corrections of the moments,
# simply choose the desired power:
power = 4

jd.M_formula(power = power)
