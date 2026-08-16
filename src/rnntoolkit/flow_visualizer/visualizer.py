import pygame
import torch
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder import FlowFieldFinder
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from rnntoolkit.flow_visualizer.visualizer_base import FlowFieldVisualizerBase

pygame.init()


class FlowFieldVisualizer(FlowFieldVisualizerBase):
    """Interactive two-dimensional viewer for an RNN's flow field.

    This class asks ``FlowFieldFinder`` to project hidden states and calculate
    motion, maintains the coordinate view during pan/zoom operations, and draws
    both the vector field and its controls. Expensive flow results are cached
    until navigation or a setting marks them as dirty.
    """

    def __init__(
        self,
        rnn,
        num_points: int = 10,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: float = 0.0,
        y_center: float = 0.0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
    ):
        super().__init__(
            rnn,
            num_points,
            x_offset,
            y_offset,
            x_center,
            y_center,
            fit_states,
            axes,
            flow_type,
        )

    def build_finder(self):
        """Build the RNNToolkit finder for this visualizer."""
        finder = FlowFieldFinder(
            rnn=self.rnn,
            num_points=self.num_points,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            x_center=self.x_center,
            y_center=self.y_center,
            fit_states=self.fit_states,
            axes=self.axes,
            follow_traj=False,
        )
        return finder

    def prepare_data(self, inputs, states, delta_inputs=None):
        """Flatten RNNToolkit inputs and states into page-aligned samples."""
        if delta_inputs is not None:
            delta_inp_nxd = FlowFieldFinderBase._nxd(delta_inputs)
        else:
            delta_inp_nxd = None
        return (
            FlowFieldFinderBase._nxd(inputs),
            FlowFieldFinderBase._nxd(states),
            delta_inp_nxd,
        )

    def compute_flow_field(self, inp_nxd, states_nxd, delta_inp_nxd=None) -> FlowField:
        """Compute one page through the finder's public flow methods."""
        state_n = states_nxd[self.current_element_idx]
        inp_n = inp_nxd[self.current_element_idx]
        if delta_inp_nxd is not None:
            delta_inp_n = delta_inp_nxd[self.current_element_idx]
        else:
            delta_inp_n = None

        finder = self.current_field()
        finder.num_points = self.preferences["grid_points"]
        finder.x_offset = self.view_span / 2.0
        finder.y_offset = self.view_span / 2.0
        # Keep the finder grid exactly aligned with the viewport. Snapping
        # this center causes gaps or apparent heatmap motion after panning.
        finder.x_center = (self.x_bounds[0] + self.x_bounds[1]) / 2.0
        finder.y_center = (self.y_bounds[0] + self.y_bounds[1]) / 2.0

        if state_n.dim() == 1:
            state_n = state_n.unsqueeze(0)
        if inp_n.dim() == 1:
            inp_n = inp_n.unsqueeze(0)

        with torch.no_grad():
            if self.flow_type == "linear":
                if delta_inp_n is None:
                    delta_inp_n = torch.zeros_like(inp_n)
                if delta_inp_n.dim() == 1:
                    delta_inp_n = delta_inp_n.unsqueeze(0)
                flow = finder.find_linear_flow(state_n, inp_n, delta_inp_n)[0]
            else:
                flow = finder.find_nonlinear_flow(state_n, inp_n)[0]

        self._flow_cache = {
            "grid": flow.grid.detach().cpu().numpy(),
            "x_vel": flow.x_vels.detach().cpu().numpy(),
            "y_vel": flow.y_vels.detach().cpu().numpy(),
            "speed": flow.speeds.detach().cpu().numpy(),
        }
        self._flow_dirty = False
        return flow
