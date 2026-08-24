import pytest
import torch
from ml.training.losses import focal_loss, multitask_loss, sequence_focal_loss


def test_focal_loss_backward():
    logits = torch.randn(8, 4, requires_grad=True)
    labels = torch.randint(0, 4, (8,))
    loss = focal_loss(logits, labels, gamma=2.0, label_smoothing=0.05)
    loss.backward()
    assert float(loss) > 0
    assert logits.grad is not None


def test_multitask_loss_modes():
    logits = [torch.randn(8, 3 if i == 0 else 4, requires_grad=True) for i in range(7)]
    labels = torch.randint(0, 3, (8, 7))
    for mode in ("ce", "focal"):
        loss = multitask_loss(logits, labels, loss_type=mode, gamma=2.0)
        assert float(loss) > 0
    # Test with custom task_weights
    weighted_loss = multitask_loss(logits, labels, task_weights=[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0], loss_type="focal")
    assert float(weighted_loss) > 0



def test_sequence_focal_loss():
    logits = torch.randn(2, 6, 50, requires_grad=True)
    labels = torch.tensor([[1, 5, 10, -100, -100, -100], [2, 4, 8, 12, -100, -100]])
    loss = sequence_focal_loss(logits, labels, ignore_index=-100, gamma=2.0)
    loss.backward()
    assert float(loss) > 0
    assert logits.grad is not None
