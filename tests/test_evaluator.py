import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_model,
)


class PerfectToyModel(nn.Module):
    """
    Toy model whose prediction is encoded
    in the first ECG sample.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        class_indices = (
            x[:, 0, 0]
            .long()
        )

        logits = torch.zeros(
            x.shape[0],
            4,
            device=x.device,
        )

        logits.scatter_(
            dim=1,
            index=class_indices.unsqueeze(1),
            value=10.0,
        )

        return logits


def make_perfect_dataloader() -> DataLoader:
    targets = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    inputs = torch.zeros(
        4,
        1,
        256,
    )

    inputs[:, 0, 0] = targets.float()

    dataset = TensorDataset(
        inputs,
        targets,
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )


def test_collect_predictions_returns_expected_labels():
    model = PerfectToyModel()

    dataloader = make_perfect_dataloader()

    y_true, y_pred = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert y_true.tolist() == [
        0,
        1,
        2,
        3,
    ]

    assert y_pred.tolist() == [
        0,
        1,
        2,
        3,
    ]


def test_evaluate_model_perfect_predictions():
    model = PerfectToyModel()

    dataloader = make_perfect_dataloader()

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert metrics[
        "macro_f1"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "balanced_accuracy"
    ] == pytest.approx(
        1.0
    )


def test_empty_dataloader_raises_error():
    inputs = torch.empty(
        0,
        1,
        256,
    )

    targets = torch.empty(
        0,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        inputs,
        targets,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
    )

    model = PerfectToyModel()

    with pytest.raises(ValueError):
        collect_predictions(
            model=model,
            dataloader=dataloader,
            device=torch.device("cpu"),
        )