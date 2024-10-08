.. _atomistic-models-outputs:

Standard model outputs
======================

In order for multiple simulation engines to be able to exploit atomic properties
computing by arbitrary metatensor atomistic models, we need all the models to
return data with specific metadata. If your model returns one of the output
defined in this documentation, then the model should follow the metadata
structure described here.

For other kind of output, you are free to use any relevant metadata structure,
but if multiple people are producing the same kind of outputs, they are
encouraged to come together, define the metadata they need and add a new section
to this page.


.. _energy:

Energy
^^^^^^

Energy is associated with the ``"energy"`` key in the model outputs, and must
have the following metadata:

.. list-table:: Metadata for energy output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the energy keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. The energy is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - if doing ``per_atom`` output, the sample names must be ``["system",
      "atom"]``, otherwise the sample names must be ``["system"]``.

      ``"system"`` must range from 0 to the number of systems given as input to
      the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the energy must not have any components

  * - properties
    - ``"energy"``
    - the energy must have a single property dimension named ``"energy"``, with
      a single entry set to ``0``.

.. _energy-gradients:

Energy gradients
----------------

Most of the time when writing an atomistic model compatible with metatensor,
gradients will be handled implicitly and computed by the simulation engine using
a backward pass. Additionally, it is possible for the model to support explicit,
forward mode gradients

The following gradients can be defined and requested with
``explicit_gradients``:

- **"positions"** (:math:`r_j`) gradients will contain the negative of the
  forces :math:`F_j`.

  .. math::

      \frac{\partial E}{\partial r_j} = -F_j

.. list-table:: Metadata for positions energy's gradients
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - samples
    - ``["sample", "system", "atom"]``
    - ``"sample"`` indicates which of the values samples we are taking the
      gradient of, and ``("system", "atom")`` indicates which of the atom's
      positions we are taking the gradients with respect to; i.e. :math:`j` in
      the equation above.

  * - components
    - ``"xyz"``
    - there must be a single component named ``"xyz"`` with values 0, 1, 2;
      indicating the direction of the displacement of the atom in the gradient
      samples.

- **"strain"** (:math:`\epsilon`) gradients will contain the stress
  :math:`\sigma` acting on the system, multiplied by the volume :math:`V`
  (sometimes also called the *virial* of this system)

  .. math::

    \frac{\partial E}{\partial \epsilon} = V \sigma

.. list-table:: Metadata for strain energy's gradients
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - **samples**
    - ``"sample"``
    - There is a single gradient sample dimension, ``"sample"`` indicating which
      of the values samples we are taking the gradient of.

  * - **components**
    - ``["xyz_1", "xyz_2"]``
    - Both ``"xyz_1"`` and ``"xyz_2"`` have values ``[0, 1, 2]``, and correspond
      to the two axes of the 3x3 strain matrix :math:`\epsilon`.


Energy ensemble
^^^^^^^^^^^^^^^

An ensemble of energies is associated with the ``"energy_ensemble"`` key in the
model outputs. Such ensembles are sometimes used to perform uncertainty
quantification, using multiple prediction to estimate an error on the mean
prediction.

Energy ensembles must have the following metadata:

.. list-table:: Metadata for energy ensemble output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - same as `Energy`_
    - same as `Energy`_

  * - samples
    - same as `Energy`_
    - same as `Energy`_

  * - components
    - same as `Energy`_
    - same as `Energy`_

  * - properties
    - ``"energy"``
    - the energy ensemble must have a single property dimension named
      ``"energy"``, with entries ranging from 0 to the number of members of the
      ensemble minus one.


Energy ensemble gradients
-------------------------

The gradient metadata for energy ensemble is the same as for the ``energy``
output (see `Energy gradients`_).

.. _features:

Features
^^^^^^^^

Features are numerical vectors representing a given structure or
atom/atom-centered environment in an abstract n-dimensional space. They are also
sometimes called descriptors, representations, embedding, *etc.*

Features can be computed with some analytical expression (for example `SOAP
power spectrum`_, `atom-centered symmetry functions`_, …), or learned internally
by a neural-network or a similar architecture.

.. _SOAP power spectrum: https://doi.org/10.1103/PhysRevB.87.184115
.. _Atom-centered symmetry functions: https://doi.org/10.1063/1.3553717

In metatensor atomistic models, they are associated with the ``"features"`` key
in the model outputs, and must adhere to the following metadata:

.. list-table:: Metadata for features output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the features keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. The feature is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - the samples should be named ``["system", "atom"]`` for per-atom outputs;
      or ``["system"]`` for global outputs.

      The ``"system"`` index should always be 0, and the ``"atom"`` index should
      be the index of the atom (between 0 and the total number of atoms). If
      ``selected_atoms`` is provided, then only the selected atoms for each
      system should be part of the samples.

  * - components
    -
    - the features must not have any components.

  * - properties
    -
    - the features can have arbitrary properties.

.. note::
  Features are typically handled without a unit, so the ``"unit"`` field of
  :py:func:`metatensor.torch.atomistic.ModelOutput` is mainly left empty.

Features gradients
------------------

As for the :ref:`energy <energy-gradients>`, features are typically used with
automatic gradient differentiation. Explicit gradients could be allowed if you
have a use case for them, but are currently not until they are fully specified.
