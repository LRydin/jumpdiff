Jump-diffusion processes
========================

We will show here how to: (1) generate trajectories of jump-diffusion processes; (2) retrieve the parameters from a single trajectory of a jump-diffusion process.
Naturally, if we already had some data -- maybe from a real-world recording of a stochastic process -- we would simply look at estimating the parameters for this process.

The theory
----------
Jump-diffusion processes\ :sup:`1`, as the name suggest, are a mixed type of stochastic processes with a diffusive and a jump term.
One form of these processes which is mathematically traceable is given by the `Stochastic Differential Equation <https://en.wikipedia.org/wiki/Stochastic_differential_equation>`_

.. math::
   \mathrm{d} X(t) = a(x,t)\;\mathrm{d} t + b(x,t)\;\mathrm{d} W(t) + \xi\;\mathrm{d} J(t),

which has four main elements: a drift term :math:`a(x,t)`, a diffusion term :math:`b(x,t)`, linked with a Wiener process :math:`W(t)`, a jump amplitude term :math:`\xi(x,t)`, which is given by a Gaussian distribution :math:`\mathcal{N}(0,\sigma_\xi^2)` coupled with a jump rate :math:`\lambda`, which is the rate of the Poissonian jumps :math:`J(t)`.
You can find a good review on this topic in Ref. 2.

Integrating a jump-diffusion process
------------------------------------
Let us use the functions in :code:`jumpdiff` to generate a jump-difussion process, and subsequently retrieve the parameters. This is a good way to understand the usage of the integrator and the non-parametric retrieval of the parameters.

First we need to load our library. We will call it :code:`jd`

.. code-block:: python
   :linenos:

   import jumpdiff as jd

Let us thus define a jump-diffusion process and use :code:`jd_process` to integrate it. Do notice here that we need the drift :math:`a(x,t)` and diffusion :math:`b(x,t)` as functions.

.. code-block:: python
   :linenos:
   :lineno-start: 2

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

   # and simply call the integration function
   X = jd.jd_process(t_final, delta_t, a=a, b=b, xi=xi, lamb=lamb)


This will generate a jump diffusion process :code:`X` of length :code:`int(10000/0.001)` with the given parameters.

.. image:: /_static/X_trajectory.png
  :height: 250
  :align: center
  :alt: A jump-difussion process


Using :code:`jumpdiff` to retrieve the parameters
-------------------------------------------------
Moments and Kramers─Moyal coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Take the timeseries :code:`X` and use the function :code:`moments` to retrieve the conditional moments of the process.
For now let us focus on the shortest time lag, so we can best approximate the Kramers---Moyal coefficients.
For this case we can simply employ

.. code-block:: python
   :linenos:
   :lineno-start: 20

   edges, moments = jd.moments(timeseries = X)

In the array :code:`edges` are the limits of our space, and in our array :code:`moments` are recorded all 6 powers/order of our conditional moments.
Let us take a look at these before we proceed, to get acquainted with them.

We can plot the first moment with any conventional plotter, so lets use here :code:`plotly` from :code:`matplotlib`.
To visualise the first moment, simply use

.. code-block:: python
   :linenos:
   :lineno-start: 21

   import matplotlib.pyplot as plt
   plt.plot(edges, moments[1]/delta_t)

.. image:: /_static/1_moment.png
  :height: 250
  :align: center
  :alt: The 1st Kramers---Moyal coefficient

The first moment here (i.e., the first Kramers---Moyal coefficient) is given solely by the drift term that we have selected :code:`-0.5*x`.
In the plot we have also included the theoretical curve, which we know from having selected the value of :code:`a(x)` in line :code:`8`.

Similarly, we can extract the second moment (i.e., the second Kramers---Moyal coefficient) is a mixture of both the contributions of the diffusive term :math:`b(x)` and the jump terms :math:`\xi` and :math:`\lambda`.

.. image:: /_static/2_moment.png
  :height: 250
  :align: center
  :alt: The 2nd Kramers---Moyal coefficient

You have this stored in :code:`moments[2]`.

Retrieving the jump-related terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Naturally one of the most pertinent questions when addressing jump-diffusion processes is the possibility of recovering these same parameters from data. For the given jump-diffusion process we can use the :code:`jump_amplitude` and :code:`jump_rate` functions to non-parametrically estimate the jump amplitude :math:`\xi` and :math:`\lambda` terms.

After having the :code:`moments` in hand, all we need is

.. code-block:: python
   :linenos:
   :lineno-start: 23

   # first estimate the jump amplitude
   xi_est = jd.jump_amplitude(moments = moments)

   # and now estimated the jump rate
   lamb_est = jd.jump_rate(moments = moments)

which resulted in our case in :code:`(xi_est) ξ = 2.43 ± 0.17` and :code:`(lamb_est) λ = 1.744 * delta_t` (don't forget to divide :code:`lamb_est` by :code:`delta_t`)!
We can compare these with our chose values in lines :code:`15-16`.
