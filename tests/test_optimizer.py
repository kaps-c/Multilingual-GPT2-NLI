from pathlib import Path

import numpy as np
import torch

from multilingual_gpt2.optimizer import AdamW


def test_optimizer_matches_preserved_reference():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2, bias=False)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1000):
        optimizer.zero_grad()
        inputs = torch.tensor(rng.uniform(size=3), dtype=torch.float32)
        target = torch.tensor([inputs[0] + inputs[1], -inputs[2]])
        ((model(inputs) - target) ** 2).sum().backward()
        optimizer.step()
    path = Path(__file__).resolve().parents[1] / "optimizer_test.npy"
    expected = torch.tensor(np.load(path))
    torch.testing.assert_close(model.weight, expected, atol=1e-6, rtol=1e-4)
