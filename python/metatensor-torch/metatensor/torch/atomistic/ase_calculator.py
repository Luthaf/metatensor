import os
from typing import Union

import numpy as np
import torch

from .. import Labels, TensorBlock
from . import MetatensorAtomisticModule, ModelOutput, ModelRunOptions, System


import ase.neighborlist  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    Calculator,
    CalculatorSetupError,
    InputError,
    PropertyNotImplementedError,
    all_properties,
)


# import here to get an error early if the user is missing metatensor-operations
from .. import sum_over_samples_block  # isort: skip


class MetatensorCalculator(Calculator):
    """TODO"""

    def __init__(
        self,
        model: Union[str, torch.jit.RecursiveScriptModule, MetatensorAtomisticModule],
        check_consistency=False,
    ):
        """TODO"""
        super().__init__()

        if isinstance(model, str):  # TODO: accept other path types here
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exists")

            self._model = torch.jit.load(model)
        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "MetatensorAtomisticModule":
                raise InputError(
                    "torch model must be 'MetatensorAtomisticModule', "
                    f"got '{model.original_name}' instead"
                )
            self._model = model
        elif isinstance(model, MetatensorAtomisticModule):
            self._model = model
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters = {
            "model": model,
            "check_consistency": check_consistency,
        }

        # We do our own check to verify if a property is implemented in calculate
        self.implemented_properties = all_properties

    def todict(self):
        # used by ASE to save the calculator
        raise NotImplementedError("todict is not yet implemented")

    @classmethod
    def fromdict(cls, dict):
        # used by ASE to load a saved calculator
        raise NotImplementedError("fromdict is not yet implemented")

    def calculate(self, atoms, properties, system_changes):
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        capabilities = self._model.capabilities()

        # check that the model can actually compute what ASE need
        outputs = _ase_properties_to_metatensor_outputs(properties)
        for name in outputs.keys():
            if name not in capabilities.outputs:
                # TODO: might not be required?
                raise PropertyNotImplementedError(
                    f"this model can not compute '{name}', the implemented "
                    f"outputs are '{[p for p in capabilities.output.keys()]}'"
                )

        # check that the model can handle the species in `atoms`
        atoms_species = set(atoms.numbers)
        for species in atoms_species:
            if species not in capabilities.species:
                raise CalculatorSetupError(
                    f"this model can not run for the atomic species '{species}'"
                )

        positions = torch.from_numpy(atoms.positions).reshape(-1, 3, 1)

        if np.all(atoms.pbc):
            cell = torch.from_numpy(atoms.cell[:]).reshape(1, 3, 3, 1)
        else:
            # TODO: handle partial pbc
            cell = torch.zeros((3, 3), dtype=torch.float64)

        do_backward = False
        if "forces" in properties:
            do_backward = True
            positions.requires_grad_(True)

        if "stress" in properties:
            scaling = torch.eye(3, requires_grad=True, dtype=cell.dtype)

            positions = positions.reshape(-1, 3) @ scaling
            positions = positions.reshape(-1, 3, 1)
            positions.retain_grad()

            cell = cell.reshape(3, 3) @ scaling
            cell = cell.reshape(1, 3, 3, 1)
            do_backward = True

        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        # convert from ase.Atoms to metatensor.torch.atomistic.System
        positions = TensorBlock(
            values=positions,
            samples=Labels(
                ["atom", "species"],
                torch.IntTensor([(i, s) for i, s in enumerate(atoms.numbers)]),
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("position", 1),
        )

        cell = TensorBlock(
            values=cell.reshape(1, 3, 3, 1),
            samples=Labels.single(),
            components=[Labels.range("cell_abc", 3), Labels.range("xyz", 3)],
            properties=Labels.range("cell", 1),
        )

        system = System(positions, cell)

        # Compute the neighbors lists requested by the model using ASE NL
        for options in self._model.requested_neighbors_lists():
            system.add_neighbors_list(
                options, neighbors=_compute_ase_neighbors(atoms, options)
            )

        run_options = ModelRunOptions()
        run_options.length_unit = "angstrom"
        run_options.selected_atoms = None
        run_options.outputs = outputs

        outputs = self._model(
            system,
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs["energy"]

        if run_options.outputs["energy"].per_atom:
            assert energy.values.shape == (len(atoms), 1)
            assert energy.samples == positions.samples
            energies = energy
            energy = sum_over_samples_block(energy, sample_names=["atom", "species"])
        else:
            assert energy.values.shape == (1, 1)

        assert len(energy.gradients_list()) == 0

        self.results = {}

        if "energies" in properties:
            self.results["energies"] = (
                energies.values.detach().to(device="cpu").numpy().reshape(-1)
            )

        assert energy.values.shape == (1, 1)
        if "energy" in properties:
            self.results["energy"] = (
                energy.values.detach().to(device="cpu").numpy()[0, 0]
            )

        if do_backward:
            energy.values.backward(-torch.ones_like(energy.values))

        if "forces" in properties:
            self.results["forces"] = (
                positions.values.grad.to(device="cpu").numpy().reshape(-1, 3)
            )

        if "stress" in properties:
            volume = atoms.cell.volume
            scaling_grad = -scaling.grad.to(device="cpu").numpy().reshape(3, 3)
            self.results["stress"] = scaling_grad / volume


def _ase_properties_to_metatensor_outputs(properties):
    energy_properties = []
    for p in properties:
        if p in ["energy", "energies", "forces", "stress", "stresses"]:
            energy_properties.append(p)

    # handle all other properties
    for p in properties:
        if p not in ["energy", "energies", "forces", "stress", "stresses"]:
            # TODO: we will want to add support for the other properties later
            raise PropertyNotImplementedError(
                f"property '{p}' it not yet supported by this calculator, "
                "even if it might be supported by the model"
            )

    output = ModelOutput()
    output.quantity = "energy"
    output.unit = "ev"
    output.forward_gradients = []

    if "energies" in properties or "stresses" in properties:
        output.per_atom = True
    else:
        output.per_atom = False

    if "stresses" in properties:
        output.forward_gradients = ["cell"]

    return {"energy": output}


def _compute_ase_neighbors(atoms, options):
    nl = ase.neighborlist.NeighborList(
        cutoffs=[options.cutoff] * len(atoms),
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=options.full_list,
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    nl.update(atoms)

    cell = torch.from_numpy(atoms.cell[:])
    positions = torch.from_numpy(atoms.positions)

    samples = []
    distances = []
    cutoff2 = options.cutoff * options.cutoff
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            distance = positions[j] - positions[i] + offset.dot(cell)

            distance2 = torch.dot(distance, distance).item()

            if distance2 > cutoff2:
                continue

            samples.append((i, j, offset[0], offset[1], offset[2]))
            distances.append(distance.to(dtype=torch.float64))

    samples = torch.tensor(samples, dtype=torch.int32)
    distances = torch.vstack(distances)

    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )
