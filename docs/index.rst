.. compass documentation master file, created by
   sphinx-quickstart on Tue Mar 22 13:13:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to compass's documentation!
======================================

.. toctree::
   :maxdepth: 4
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

Code
----
To get the odds ratios of all candidates use the ``Survey`` class:

.. code-block:: python

   import pandas as pd
   from compass import model
   from compass import helperfunctions
   
   observation = pd.read_csv('observation.csv')
   survey_object = model.Survey(observation, magnitudes_column_name)
   survey_object.set_fieldstar_models(
       # Color transformed column name from Gaias G-Band.
       magnitudes_column_name_CALC,
       # Column name of the corresponding magnitude in 2MASS.
       magnitudes_column_name_2MASS,
       cone_radius=0.3,  # in degree
       binsize=200  # Number of objects in a single magnitude bin
   )
   # Inflating parameters to adjust the sharp drop-off of the Gaussians.
   
   survey_object.set_evaluated_fieldstar_models(
       sigma_cc_min=0,
       sigma_model_min=0
   )

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

.. automodapi:: compass.model
   :no-inheritance-diagram:
.. automodapi:: compass.helperfunctions
.. automodapi:: compass.preset_plots
