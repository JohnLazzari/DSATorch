import torch
from torch.nn import RNN, GRU, LSTM
import torch.nn.functional as F
import numpy as np


class Linearization:
    def __init__(
        self,
        rnn: RNN | GRU | LSTM,
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        Args:
            mrnn: mRNN object
            W_inp: Custom input weights to be used when linearizing
            W_rec: Custom recurrent weights to be used when linearizing
        """
        self.rnn = rnn

    @staticmethod
    def relu_grad(x: torch.Tensor) -> torch.Tensor:
        """
        relu function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.relu, x)

    @staticmethod
    def tanh_grad(x: torch.Tensor) -> torch.Tensor:
        """
        tanh function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.tanh, x)

    def jacobian(
        self, h: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Linearize the dynamics around a state and return the Jacobian.

        Computes the Jacobian of the mRNN update with respect to the hidden state
        evaluated at the provided state ``x`` and (optionally) a subset of regions
        defined by ``*args``. If ``W_inp`` is provided, also returns the Jacobian
        with respect to the input.

        Args:
            x (torch.Tensor): 1D or batched tensor representing the pre-activation state at which to
                linearize (shape ``[H]``).
            *args (str): Optional region names specifying a subset for the Jacobian.
            alpha (float): Discretization factor used in the update.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Jacobian w.r.t. hidden
            state, and optionally (Jacobian w.r.t. input) if ``W_inp`` is provided.
        """
        assert h.dim() == 1

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        # Implementing h'(x), diagonalize to multiply by W
        if self.rnn.nonlinearity == "relu":
            d_x_act_diag = self.relu_grad(h)
        elif self.rnn.nonlinearity == "tanh":
            d_x_act_diag = self.tanh_grad(h)
        else:
            raise ValueError("not a valid activation function")

        # Get final jacobian using form above
        _jacobian = d_x_act_diag @ self.rnn.weight_hh_l0
        _jacobian_inp = d_x_act_diag @ self.rnn.weight_ih_l0
        return _jacobian, _jacobian_inp

    def eigendecomposition(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearize the network and compute eigen decomposition.

        Args:
            x (torch.Tensor): 1D hidden state where the system is linearized.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        _jacobian, _ = self.jacobian(x)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        # Split real and imaginary parts
        reals = []
        for eigenvalue in eigenvalues:
            reals.append(eigenvalue.real.item())
        reals = torch.tensor(reals)

        ims = []
        for eigenvalue in eigenvalues:
            ims.append(eigenvalue.imag.item())
        ims = torch.tensor(ims)

        return reals, ims, eigenvectors
