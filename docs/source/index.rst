jumpdiff
========

.. toctree::
   :maxdepth: 3


:code:`jumpdiff` is a :code:`python` library with non-parametric Nadaraya---Watson estimators to extract the parameters of jump-diffusion processes.
With :code:`jumpdiff` one can extract the parameters of a jump-diffusion process from one-dimensional timeseries, employing both a kernel-density estimation method combined with a set on second-order corrections for a precise retrieval of the parameters for short timeseries.

.. include:: installation.rst

.. include:: jd_processes.rst

.. include:: q_ratio_exp.rst

Table of Content
================

.. toctree::
   :maxdepth: 3

   installation
   jd_processes
   q_ratio_exp
   functions/index
   license

Literature
==========

| :sup:`1` Tabar, M. R. R. *Analysis and Data-Based Reconstruction of Complex Nonlinear Dynamical Systems.* Springer, International Publishing (2019), Chapter `Stochastic Processes with Jumps and Non-vanishing Higher-Order Kramers---Moyal Coefficients* <https://doi.org/10.1007/978-3-030-18472-8_11>`_.
| :sup:`2` Friedrich, R., Peinke, J., Sahimi, M., Tabar, M. R. R. *Approaching complexity by stochastic methods: From biological systems to turbulence,* `Physics Reports 506, 87–162 (2011) <https://doi.org/10.1016/j.physrep.2011.05.003>`_.
| :sup:`3` Rydin Gorjão, L., Meirinhos, F. *kramersmoyal: Kramers–Moyal coefficients for stochastic processes.* `Journal of Open Source Software, 4(44) (2019) <https://doi.org/10.21105/joss.01693>`_.

An extensive review on the subject can be found `here <http://sharif.edu/~rahimitabar/pdfs/80.pdf>`_.

Funding
=======

Helmholtz Association Initiative *Energy System 2050 - A Contribution of the Research Field Energy* and the grant No. VH-NG-1025 and *STORM - Stochastics for Time-Space Risk Models* project of the Research Council of Norway (RCN) No. 274410.
