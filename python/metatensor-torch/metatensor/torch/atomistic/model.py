from typing import Dict, List, Optional

import torch

from .. import TensorBlock
from . import ModelCapabilities, ModelRunOptions, NeighborsListOptions, System
from .units import KNOWN_QUANTITIES, Quantity


class MetatensorAtomisticModule(torch.nn.Module):
    """TODO"""

    # Some annotation to make the TorchScript compiler happy
    _requested_neighbors_lists: List[NeighborsListOptions]
    _known_quantities: Dict[str, Quantity]

    def __init__(self, module: torch.nn.Module, capabilities: ModelCapabilities):
        """TODO"""
        super().__init__()

        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"`module` should be a torch.nn.Module, not {type(module)}")

        if isinstance(module, torch.jit.RecursiveScriptModule):
            raise TypeError("module should not already be a ScriptModule")

        if module.training:
            raise ValueError("module should not be in training mode")

        _check_annotation(module)
        self._module = module

        # ============================================================================ #

        # recursively explore `module` to get all the requested_neighbors_lists
        self._requested_neighbors_lists = []
        _get_requested_neighbors_lists(
            self._module,
            self._module.__class__.__name__,
            self._requested_neighbors_lists,
        )
        # ============================================================================ #

        self._capabilities = capabilities
        self._known_quantities = KNOWN_QUANTITIES

        length = self._known_quantities["length"]
        length.check_unit(self._capabilities.length_unit)

        # Check the units of the outputs
        for name, output in self._capabilities.outputs.items():
            if output.quantity == "":
                continue

            if output.quantity not in self._known_quantities:
                raise ValueError(
                    f"unknown output quantity '{output.quantity}' for '{name}' output, "
                    f"only {list(self._known_quantities.keys())} are supported"
                )

            quantity = self._known_quantities[output.quantity]
            quantity.check_unit(output.unit)

    @torch.jit.export
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @torch.jit.export
    def requested_neighbors_lists(
        self,
        length_unit: Optional[str] = None,
    ) -> List[NeighborsListOptions]:
        if length_unit is not None:
            length = self._known_quantities["length"]
            conversion = length.conversion(self._capabilities.length_unit, length_unit)
        else:
            conversion = 1.0

        result: List[NeighborsListOptions] = []
        for request in self._requested_neighbors_lists:
            new_request = NeighborsListOptions(
                cutoff=request.cutoff * conversion,
                full_list=request.full_list,
            )

            for requestor in request.requestors():
                new_request.add_requestor(requestor)

            result.append(new_request)

        return result

    def forward(
        self,
        system: System,
        run_options: ModelRunOptions,
        check_consistency: bool,
    ) -> Dict[str, TensorBlock]:
        """TODO"""

        if check_consistency:
            _check_consistency(self._capabilities, run_options)
            # TODO: check we got the right set of neighbors lists in `system`

        # convert systems from engine to model units
        if self._capabilities.length_unit != run_options.length_unit:
            length = self._known_quantities["length"]
            conversion = length.conversion(
                from_unit=run_options.length_unit,
                to_unit=self._capabilities.length_unit,
            )

            system.positions.values[:] *= conversion
            system.cell.values[:] *= conversion

            # also update the neighbors list distances
            for options in self._requested_neighbors_lists:
                neighbors = system.get_neighbors_list(options)
                neighbors.values[:] *= conversion

        # run the actual calculations
        outputs = self._module(system=system, run_options=run_options)

        # convert outputs from model to engine units
        for name, output in outputs.items():
            declared = self._capabilities.outputs[name]
            requested = run_options.outputs[name]
            if declared.quantity == "" or requested.quantity == "":
                continue

            if declared.quantity != requested.quantity:
                raise ValueError(
                    f"model produces values as '{declared.quantity}' for the '{name}' "
                    f"output, but the engine requested '{requested.quantity}'"
                )

            quantity = self._known_quantities[declared.quantity]
            output.values[:] *= quantity.conversion(
                from_unit=declared.unit, to_unit=requested.unit
            )

        return outputs

    def export(self, file):
        """TODO"""

        module = self.eval()
        try:
            module = torch.jit.script(module)
        except RuntimeError as e:
            raise RuntimeError("could not convert the module to TorchScript") from e

        # TODO: can we freeze these?
        # module = torch.jit.freeze(module)

        # TODO: record torch version

        # TODO: record list of loaded extensions

        torch.jit.save(module, file, _extra_files={})


def _get_requested_neighbors_lists(
    module: torch.nn.Module,
    name: str,
    requested: List[NeighborsListOptions],
):
    if hasattr(module, "requested_neighbors_lists"):
        for new_options in module.requested_neighbors_lists():
            new_options.add_requestor(name)

            already_requested = False
            for existing in requested:
                if existing == new_options:
                    already_requested = True
                    for requestor in new_options.requestors():
                        existing.add_requestor(requestor)

            if not already_requested:
                requested.append(new_options)

    for child_name, child in module.named_children():
        _get_requested_neighbors_lists(child, name + "." + child_name, requested)


def _check_annotation(module: torch.nn.Module):
    # check annotations on forward
    annotations = module.forward.__annotations__
    expected_arguments = [
        "system",
        "run_options",
        "return",
    ]

    expected_signature = (
        "`forward(system: System, run_option: ModelRunOptions) -> "
        "Dict[str, TensorBlock]`"
    )

    if list(annotations.keys()) != expected_arguments:
        raise TypeError(
            "`module.forward()` takes unexpected arguments, expected signature is "
            + expected_signature
        )

    if annotations["system"] != System:
        raise TypeError(
            "`system` argument must be a metatensor atomistic `System`, "
            f"not {annotations['system']}"
        )

    if annotations["run_options"] != ModelRunOptions:
        raise TypeError(
            "`run_options` argument must be a metatensor atomistic `ModelRunOptions`, "
            f"not {annotations['run_options']}"
        )

    if annotations["return"] != Dict[str, TensorBlock]:
        raise TypeError(
            "`forward()` must return a `Dict[str, TensorBlock]`, "
            f"not {annotations['return']}"
        )


def _check_consistency(capabilities: ModelCapabilities, run_options: ModelRunOptions):
    for name, requested in run_options.outputs.items():
        if name not in capabilities.outputs:
            raise ValueError(
                f"this model can not compute the requested '{name}' output"
            )

        output = capabilities.outputs[name]

        for gradient in requested.forward_gradients:
            if gradient not in output.forward_gradients:
                raise ValueError(
                    f"this model can not compute gradients w.r.t. '{gradient}' in "
                    + f"forward mode for the '{name}' output"
                )

        if requested.per_atom and not output.per_atom:
            raise ValueError(f"this model can not compute the '{name}' output per-atom")
