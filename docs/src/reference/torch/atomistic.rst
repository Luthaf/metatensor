.. _atomistic-models:

Atomistic models
================

A first set of classes in this module is used to define the input of atomistic
models:

.. autoclass:: metatensor.torch.atomistic.System
    :members:

.. autoclass:: metatensor.torch.atomistic.NeighborsListOptions
    :members:
    :special-members: __eq__, __ne__

--------------------------------------------------------------------------------

The second set of classes in this module is used to define the capacities of a
given model, and request a specific calculation.

.. autoclass:: metatensor.torch.atomistic.ModelOutput
    :members:

.. autoclass:: metatensor.torch.atomistic.ModelCapabilities
    :members:

.. autoclass:: metatensor.torch.atomistic.ModelRunOptions
    :members:
