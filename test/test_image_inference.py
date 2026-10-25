# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Portable image inference. Set SAM3_TEST_CHECKPOINT to also exercise real weights."""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.vitdet import Mlp
from sam3.model_builder import build_sam3_image_model


DEVICES = [
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS is unavailable"
        ),
    ),
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
]


@pytest.fixture(scope="module")
def image_model():
    torch.manual_seed(42)
    checkpoint = os.environ.get("SAM3_TEST_CHECKPOINT")
    model = build_sam3_image_model(
        device="cpu", checkpoint_path=checkpoint, load_from_HF=False
    )
    if checkpoint is None:
        # Upstream leaves these parameters uninitialized until checkpoint loading.
        # Initialize them so weight-free smoke tests cannot pass via NaN filtering.
        with torch.no_grad():
            text = model.backbone.language_backbone.encoder
            text.positional_embedding.normal_(std=0.01)
            text.text_projection.normal_(std=0.01)
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_single_image_text_and_box_prompts(image_model, device):
    model = image_model.to(torch.device(device))
    processor = Sam3Processor(model, confidence_threshold=0.0)
    assert processor.device == model.device
    pixels = np.arange(96 * 160 * 3, dtype=np.uint8).reshape(96, 160, 3)
    state = processor.set_image(Image.fromarray(pixels))

    def check_predictions(output):
        count = len(output["scores"])
        assert count > 0  # An empty result could hide nonfinite model outputs.
        assert output["boxes"].shape == (count, 4)
        assert output["masks"].shape == (count, 1, 96, 160)
        assert output["masks"].dtype == torch.bool
        for key in ("boxes", "scores", "masks_logits"):
            assert output[key].device == model.device
            assert torch.isfinite(output[key]).all()
        assert ((output["scores"] >= 0) & (output["scores"] <= 1)).all()

    output = processor.set_text_prompt("shoe", state)
    check_predictions(output)
    scores = output["scores"].clone()
    processor.reset_all_prompts(state)
    output = processor.set_text_prompt("shoe", state)
    torch.testing.assert_close(output["scores"], scores)

    # Box-only prompting must not depend on a previous text prompt.
    processor.reset_all_prompts(state)
    check_predictions(processor.add_geometric_prompt([0.6, 0.4, 0.3, 0.2], True, state))
    check_predictions(
        processor.add_geometric_prompt([0.2, 0.3, 0.1, 0.2], False, state)
    )
    output = processor.set_confidence_threshold(1.0, state)
    assert output["boxes"].shape == (0, 4)
    assert output["masks"].shape == (0, 1, 96, 160)
    assert output["scores"].shape == (0,)


@pytest.mark.parametrize("device", DEVICES)
def test_mlp_preserves_float32_results(device):
    torch.manual_seed(7)
    mlp = Mlp(6, hidden_features=10, out_features=4).eval()
    inputs = torch.linspace(-2, 3, 18).reshape(3, 6)
    expected = F.linear(
        F.gelu(F.linear(inputs, mlp.fc1.weight, mlp.fc1.bias)),
        mlp.fc2.weight,
        mlp.fc2.bias,
    )
    with torch.inference_mode():
        actual = mlp.to(device)(inputs.to(device)).cpu()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_position_encoding_cache_follows_input_device(device):
    encoding = PositionEmbeddingSine(16, precompute_resolution=112)
    # 8x8 is precomputed; 5x9 also exercises an uncached, asymmetric shape.
    for height, width in [(8, 8), (5, 9)]:
        inputs = torch.zeros(2, 1, height, width)
        expected = PositionEmbeddingSine(16)(inputs)
        actual = encoding(inputs.to(device))
        assert actual.device.type == torch.device(device).type
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
        # Moving back must not reuse a tensor on the previous accelerator.
        torch.testing.assert_close(encoding(inputs), expected)
