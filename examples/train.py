"""
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Model
from flip_flop_data import FlipFlopData


def train_FlipFlop():
    """Train an RNN to solve the N-bit memory task.
    Args:
        None.
    Returns:
        model: FlipFlop object.
                The trained RNN model.
        valid_predictions: dict.
                The model's predictions on a set of held-out validation trials.
    """

    # Data specifications
    n_bits = 3
    n_train = 512
    epochs = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model hyperparameters
    n_hidden = 16

    data_gen = FlipFlopData()
    train_data = data_gen.generate_data(n_trials=n_train)

    model = Model(n_bits, n_hidden, n_bits).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        running_loss = 0
        optimizer.zero_grad()
        # Gather inputs and targets
        input = torch.from_numpy(train_data["inputs"]).to(device)
        target = torch.from_numpy(train_data["targets"]).to(device)

        # forward pass through model
        h0 = torch.zeros(size=(1, n_train, n_hidden)).to(device)
        out, hn = model(input, h0)

        # loss and update parameters
        loss_ = criterion(out, target)
        loss_.backward()
        optimizer.step()

        running_loss += loss_.item()

        if epoch % 10 == 0:
            running_loss /= 10
            print(f"Epoch: {epoch}, Avg. Loss: {running_loss}")
            running_loss = 0

        # Save model
        torch.save(
            model.state_dict(),
            "flip_flop_rnn.pth",
        )


def main():
    # Step 1: Train an RNN to solve the N-bit memory task
    train_FlipFlop()


if __name__ == "__main__":
    main()
