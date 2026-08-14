"""Visualize a three-neuron RNN line attractor.

The first neuron is a neutral (integrator) direction and the other two are
stable contracting directions. A constant input moves the state along the line attractor.
"""

import numpy as np
import torch
import torch.nn as nn

from rnntoolkit import FlowFieldVisualizer


def make_line_attractor():
    """Create h[t+1] = ReLU(W_hh h[t] + W_ih u[t]).

    With W_hh[0, 0] = 1, the first neuron is the line-attractor/integrator
    coordinate. The other two neurons contract toward zero.
    """
    rnn = nn.RNN(input_size=1, hidden_size=3, batch_first=True, nonlinearity="relu")
    with torch.no_grad():
        rnn.weight_ih_l0.zero_()
        rnn.weight_ih_l0[0, 0] = 1.0
        rnn.weight_hh_l0.zero_()
        rnn.weight_hh_l0[0, 0] = 1.0
        rnn.weight_hh_l0[1, 1] = 0.8
        rnn.weight_hh_l0[2, 2] = 0.7
        rnn.bias_ih_l0.zero_()
        rnn.bias_hh_l0.zero_()
    return rnn


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    rnn = make_line_attractor()
    rnn.eval()

    # Three trials share the same static positive input. The first neuron
    # integrates it, while different initial states provide batch variation
    # for the visualizer's PCA projection.
    n_trials = 3
    n_time = 150
    static_input = 0.02
    inputs = torch.full((n_trials, n_time, 1), static_input)
    h0 = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.4, 0.1]],
            [[2.0, 0.8, 0.2]],
        ]
    ).transpose(0, 1)

    with torch.no_grad():
        h_traj, _ = rnn(inputs, h0)

    print("inputs:", inputs.shape)
    print("hidden trajectories:", h_traj.shape)
    print("final state:", h_traj[0, -1])

    # The hidden state is three-dimensional; FlowFieldFinder fits its 2D PCA
    # projection using all samples from all three batch trajectories.
    visualizer = FlowFieldVisualizer(
        rnn,
        fit_states=h_traj,
        num_points=15,
        x_offset=3,
        y_offset=3,
        x_center=1.5,
        y_center=0,
    )
    visualizer.run(inputs, h_traj)
