"""Atom-JEPA forwards run one at a time, whatever threads the inference server calls from.

Local + fast (no AWS, no checkpoint): the encoder is replaced by a module that records
whether two forwards were ever inside it together.
"""

import threading
import time

import pytest

pytest.importorskip("atom_jepa")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from workbench.endpoints import atomjepa_utils  # noqa: E402
from workbench.endpoints.atomjepa_utils import FoldModel  # noqa: E402


class OverlapDetector(nn.Module):
    """Stands in for the fine-tuned model: a forward that notices another one in progress."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.encoder = None
        self.running = 0
        self.overlapped = False
        self.calls = 0

    def forward(self, batch: dict) -> torch.Tensor:
        self.calls += 1
        self.running += 1
        self.overlapped |= self.running > 1
        time.sleep(0.02)
        self.running -= 1
        return torch.zeros(batch["n"])


def test_concurrent_predicts_never_overlap_a_forward(monkeypatch):
    monkeypatch.setattr(atomjepa_utils, "graph_collator", lambda encoder: lambda samples: {"n": len(samples)})
    monkeypatch.setattr(atomjepa_utils, "move_batch", lambda batch, device: batch)
    model = OverlapDetector()
    fold = FoldModel(model=model, target_mean=0.0, target_std=1.0)
    conformers = [[{}], [{}], [{}]]  # three molecules, one conformer each

    results = []
    threads = [threading.Thread(target=lambda: results.append(fold.predict(conformers))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.calls == 4
    assert not model.overlapped
    assert all(len(result) == 3 for result in results)
