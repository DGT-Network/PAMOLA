PAMOLA Core API Reference
=========================

**PAMOLA Core** is an open-source Python library for privacy-preserving data
operations developed by `Realm Inveo Inc. <https://realminveo.com>`_ and
`DGT Network Inc. <https://dgt.world>`_.

Complete reference for every public symbol exported by ``pamola_core``.
All public APIs are importable directly from the top-level package.

Quick Example
-------------

.. code-block:: python

   import pamola_core as pc

   df = pc.read_csv("data.csv")
   op = pc.FullMaskingOperation(...)
   result = op.execute(df)


API Reference
-------------

.. autosummary::
   :toctree: generated

   pamola_core

----

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
