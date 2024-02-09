import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_zeros_like():
    tensor = load_data("qm7-power-spectrum.npz")
    zero_tensor = metatensor.torch.zeros_like(tensor)

    # right output type
    assert isinstance(zero_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert zero_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(zero_tensor, tensor)

    # right values
    for block in zero_tensor.blocks():
        assert torch.all(block.values == 0)


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.zeros_like, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
