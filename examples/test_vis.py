# Imports and setup
import torch
import numpy as np
from sklearn.decomposition import PCA

from flip_flop_data import FlipFlopData
from model import Model

# dsatorch imports
from rnntoolkit import FlowFieldVisualizer

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Load the trained flip-flop RNN
    ckpt_path = "flip_flop_rnn.pth"
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Model hyperparameters used in training (see train.py)
    n_bits = 3
    n_hidden = 16

    model = Model(n_bits, n_hidden, n_bits)
    model.load_state_dict(state_dict)
    model.eval()

    print(model)

    # Generate trials and collect hidden-state trajectories
    n_trials = 150
    n_time = 150

    data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time)
    data = data_gen.generate_data(n_trials=n_trials)

    inputs = torch.from_numpy(data["inputs"])

    # Initial hidden state for the RNN
    h0 = torch.zeros(1, n_trials, n_hidden)

    with torch.no_grad():
        outputs, h_traj = model(inputs, h0)  # h_traj: [batch, time, hidden]

    pca = PCA(n_components=5)
    pca.fit(h_traj.reshape(-1, n_hidden))
    comps = torch.from_numpy(pca.components_[0:2])

    print("inputs:", inputs.shape)
    print("outputs:", outputs.shape)
    print("hidden trajectories:", h_traj.shape)

    # Flow field (nonlinear) + fixed points in 2D PCA space
    fff = FlowFieldVisualizer(
        model.rnn,
        inputs,
        fit_states=h_traj,
        num_points=15,
        x_offset=3,
        y_offset=3,
        x_center=0,
        y_center=0,
        axes=comps,
    )
    fff.run()

