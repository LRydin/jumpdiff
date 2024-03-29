import numpy as np
import matplotlib.pyplot as plt

# This is a short python script to play around with jumpdiff
import jumpdiff as jd

# %% Let's first integrate a jump-diffusion process
# Define:
# integration time and time sampling
t_final = 10000
delta_t = 0.01

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
X = jd.jd_process(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb)


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

xi_est = jd.jump_amplitude(moments = moments, verbose = True)
print(xi_est)

lamb_est = jd.jump_rate(moments = moments, xi_est = xi, verbose = True)
print(lamb_est/delta_t)
# Don't forget that the jump rate λ needs to be divide by 'delta_t' to yield a
# comparible result.

# %% #########################################################################

# To understand the usage of the q_ratio function, let us generate to sample
# trajectories: one without jumps, denoted d_timeseries, and one with jumps
# denoted j_timeseries.

# integration time and time sampling
t_final = 10000
delta_t = 0.01

# Drift function
def a(x):
    return -0.5*x

# Diffusion function
def b(x):
    return 0.75

# generate 2 trajectories
d_timeseries = jd.jd_process(t_final, delta_t, a=a, b=b, xi=0, lamb=0)
j_timeseries = jd.jd_process(t_final, delta_t, a=a, b=b, xi=2.5, lamb=1.75)

# %% Subsequently we time a time scale to analyse, as
lag = np.logspace(0, 3, 25, dtype=int)

d_lag, d_Q = jd.q_ratio(lag, d_timeseries)
j_lag, j_Q = jd.q_ratio(lag, j_timeseries)

# %% we can then finally plot the results
fig, ax = plt.subplots(1,1,figsize=(6,3))

ax.loglog(d_lag, d_Q, '-', color = 'black', label='diffusion')
ax.loglog(j_lag, j_Q, 'o-', color = 'black', label='jump-diffusion')
# ax.set_xlim([-5,5]); ax.set_ylim([-5,5])

ax.set_xlabel('lag', fontsize=16)
ax.set_ylabel('$Q$-ratio', fontsize=16)
ax.legend(fontsize=13)

fig.tight_layout()
fig.savefig('q_ratio.png', dpi=300)

# %% If one wishes to check the formulae behind the corrections of the moments,
# simply choose the desired power:
power = 4

jd.m_formula(power = power)
