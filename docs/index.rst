.. compass documentation master file, created by
   sphinx-quickstart on Tue Mar 22 13:13:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to compass's documentation!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


This package presents a method to evaluate the likelihood of a candidate being a true companion to a host star using a proper motion model based on stochastic models.


Usage
=====

.. _installation:

Installation
------------

You can install `compass` by installing this repo:
.. code-block::
   pip install git+https://github.com/herzphi/compass.git


.. _code:

Code
----
To get the odds ratios of all caniddates use the ``get_p_ratio_table`` function:

.. py:function:: compass.get_p_ratio_table(target_name, cone_radius, candidates, sigma_cc_min, sigma_model_min)
   
   Return a dataframe containing the data of all caniddates and the odds ratios.
   
   :param target_name: Name of the host star.
   :type target_name: str
   :param cone_radius: Radius in degrees of the queried cone centered at the host stars position.
   :type cone_radius: float
   :param candidates: Astrometric data on the candidates.
   :type candidates: pandas.DataFrame
   :param sigma_cc_min: Inflating parameter for the caniddates likelihood in mas/yr.
   :type sigma_cc_min: float
   :param sigma_model_min: Inflating parameter for the model likelihood in mas/yr.
   :type sigma_model_min: float
   :return: Table containing the data and odds ratios of the candidates.
   :rtype: pandas.DataFrame

..
  The following section creates an index, a list of modules and a 
  search page. 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..
 The following will add the signature of the individual functions and pull
 their docstrings.

.. automodapi:: compass.modelling
