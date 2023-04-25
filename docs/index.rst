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

To use `compass` you need to install astropy, astroquery, matplotlib, numpy, pandas, requests, scipy and seaborn.


.. _code:

Code
----

Here is an example of describing the use of your function.

To add one to a number you can use the ``compass.add_one()`` function:

.. py:function:: compass.modelling.HelperFunctions.func_lin()


    :param number: Should be integer, floating point number or a string. 

    If ``number`` is not one of these types, an exception will be raised:

    .. py:exception:: TypeError

   Raised if the input is invalid.

More about how to describe code can be hound 
`here <https://www.sphinx-doc.org/en/master/tutorial/describing-code.html>`_


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
