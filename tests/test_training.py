import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.models.cnn1d import ECGCNN1D
from src.training.engine import (
    evaluate_loss,
    train_one_epoch,
)


def make_test_dataloader() -> DataLoader:
    inputs = torch.randn(
        16,
        1,
        256,
    )

    targets = torch.randint(
        low=0,
        high=4,
        size=(16,),
    )

    dataset = TensorDataset(
        inputs,
        targets,
    )

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )


def test_train_one_epoch_returns_finite_loss():
    model = ECGCNN1D()

    dataloader = make_test_dataloader()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(
        torch.tensor(loss)
    )


def test_training_updates_model_parameters():
    model = ECGCNN1D()

    dataloader = make_test_dataloader()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    assert not torch.equal(
        parameter_before,
        parameter_after,
    )


def test_evaluate_loss_returns_finite_loss():
    model = ECGCNN1D()

    dataloader = make_test_dataloader()

    criterion = nn.CrossEntropyLoss()

    loss = evaluate_loss(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(
        torch.tensor(loss)
    )


def test_evaluation_does_not_update_parameters():
    model = ECGCNN1D()

    dataloader = make_test_dataloader()

    criterion = nn.CrossEntropyLoss()

    parameter_before = (
        next(model.parameters())
        .detach()
        .clone()
    )

    evaluate_loss(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    parameter_after = (
        next(model.parameters())
        .detach()
        .clone()
    )

    assert torch.equal(
        parameter_before,
        parameter_after,
    )