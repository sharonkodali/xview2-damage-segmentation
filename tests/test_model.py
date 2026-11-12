import pytest
import torch

from mask_utils import NUM_CLASSES
from model import UNet


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_default_head_has_five_classes():
    model = UNet(base=4)
    assert model.outc.out_channels == NUM_CLASSES == 5
    assert model.inc.block[0].in_channels == 6


@pytest.mark.parametrize("bilinear", [True, False])
def test_forward_output_shape(bilinear):
    model = UNet(in_channels=6, num_classes=NUM_CLASSES, base=8, bilinear=bilinear).eval()
    x = torch.rand(2, 6, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES, 64, 64)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_non_power_of_two_input_keeps_spatial_size():
    model = UNet(base=4).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 6, 50, 70))
    assert out.shape == (1, NUM_CLASSES, 50, 70)


def test_custom_class_count():
    model = UNet(in_channels=6, num_classes=3, base=4).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 6, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_backward_with_ignore_index_produces_finite_gradients():
    model = UNet(base=4)
    x = torch.rand(2, 6, 32, 32)
    target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
    target[:, :4, :4] = 255
    loss = torch.nn.CrossEntropyLoss(ignore_index=255)(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
