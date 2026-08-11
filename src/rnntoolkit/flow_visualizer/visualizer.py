import pygame
import numpy as np
import math
import sys
import torch
from matplotlib import colormaps
from rnntoolkit.flow_fields.flow_field_finder import FlowFieldFinder
from rnntoolkit.flow_visualizer.button import Button
from rnntoolkit.flow_visualizer.drop_down import DropdownMenu
from rnntoolkit.flow_visualizer.preferences import PreferencesPanel
from rnntoolkit.flow_fields.flow_field import FlowField

pygame.init()

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 650
TOP_BAR_HEIGHT = 35
TOOLBAR_HEIGHT = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
MEDIUM_GRAY = (150, 150, 150)
DARK_GRAY = (80, 80, 80)
CANVAS_BG = (245, 245, 250)


def fmt_num(v):
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if s == "-0":
        return "0"
    return s


PREFERENCE = {
    "grid_points": {
        "min": 5,
        "max": 21,
        "step": 1,
        "fmt": lambda v: str(int(v)),
        "is_int": True,
    },
    "scroll_speed": {
        "min": 0.2,
        "max": 3.0,
        "step": 0.2,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "zoom_speed": {
        "min": 0.2,
        "max": 3.0,
        "step": 0.2,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "arrow_size_mult": {
        "min": 0.5,
        "max": 3.5,
        "step": 0.1,
        "fmt": lambda v: f"{v:.1f}x",
        "is_int": False,
    },
    "colormap": {
        "choices": ("viridis", "plasma", "inferno", "Purples", "coolwarm"),
        "fmt": str,
    },
}


class FlowFieldVisualizer:
    """Interactive two-dimensional viewer for an RNN's flow field.

    This class asks ``FlowFieldFinder`` to project hidden states and calculate
    motion, maintains the coordinate view during pan/zoom operations, and draws
    both the vector field and its controls. Expensive flow results are cached
    until navigation or a setting marks them as dirty.
    """

    def __init__(
        self,
        rnn,
        inputs: torch.Tensor,
        num_points: int = 10,
        delta_inputs: torch.Tensor | None = None,
        x_offset: int = 5,
        y_offset: int = 5,
        x_center: int = 0,
        y_center: int = 0,
        fit_states: torch.Tensor | None = None,
        axes: torch.Tensor | None = None,
        flow_type: str = "nonlinear",
    ):
        """Create the window, prepare the supplied data, and build the UI."""
        # Pygame window and frame-rate controller.
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Flow Field Visualizer")
        self.clock = pygame.time.Clock()

        # User-adjustable rendering and navigation settings.
        self.preferences = {
            "grid_points": num_points,
            "scroll_speed": 1.0,
            "zoom_speed": 1.0,
            "arrow_size_mult": 1.8,
            "colormap": "viridis",
        }
        self.small_grid = 10

        # Initial projection and grid configuration.
        self.num_points = num_points
        self.x_center = x_center
        self.y_center = y_center
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.view_span = 10.0

        # Establish the square canvas and its initial data-coordinate bounds.
        self._calculate_grid_area()
        self.x_bounds = [0.0, 10.0]
        self.y_bounds = [0.0, 10.0]
        self._update_bounds()

        # Model data consumed by FlowFieldFinder.
        self.rnn = rnn
        self.fit_states = fit_states
        self.inputs = inputs
        self.delta_inputs = delta_inputs
        self.axes = axes
        self.current_element_idx = 0

        # Each page stores a finder and the input width it expects.
        self.pages = [self._make_field_from_user_data()]

        self.current_page = 1
        self.total_pages = 1
        self.flow_type = flow_type

        if self.flow_type != "nonlinear" and self.flow_type != "linear":
            raise ValueError("Must be nonlinear or linear")

        # The vector field is computed lazily the first time it is drawn.
        self._flow_cache = None
        self._flow_dirty = True

        self._create_ui()
        self.running = True

        # State retained while the user pans with a left-mouse drag.
        self.dragging = False
        self.drag_last_pos = None

    def _make_field_from_user_data(self):
        """Build a finder and normalize tensors to sample-first form."""
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
        self.states_nxd = finder._nxd(self.fit_states)
        self.inputs_nxd = finder._nxd(self.inputs)
        if self.delta_inputs is not None:
            self.delta_inputs_nxd = finder._nxd(self.delta_inputs)
        else:
            self.delta_inputs_nxd = None

        self.input_size = self.inputs.shape[-1]

        return {"finder": finder, "input_size": self.input_size}

    def current_field(self):
        """Return metadata for the currently visible field page."""
        return self.pages[self.current_page - 1]

    def _mark_dirty(self):
        """Require the cached flow field to be recomputed before drawing."""
        self._flow_dirty = True

    def _calculate_grid_area(self):
        """Fit a centered square canvas between the two toolbars."""
        top_margin = 30
        bottom_margin = 45
        side_margin = 80

        available_width = WINDOW_WIDTH - side_margin * 2
        available_height = (
            WINDOW_HEIGHT - TOP_BAR_HEIGHT - TOOLBAR_HEIGHT - top_margin - bottom_margin
        )

        size = min(available_width, available_height)

        left = (WINDOW_WIDTH - size) // 2
        top = TOP_BAR_HEIGHT + top_margin

        self.grid_area = pygame.Rect(left, top, size, size)

    def _create_ui(self):
        """Create all buttons and popup controls."""
        self.file_btn = Button(14, 5, 70, 25, "File", style="menu")
        self.pref_btn = Button(88, 5, 105, 25, "Preferences", style="menu")
        self.file_dropdown = DropdownMenu(self.file_btn, ["Save Current", "Save All"])
        self.preferences_panel = PreferencesPanel(self.pref_btn, self)

        toolbar_y = WINDOW_HEIGHT - TOOLBAR_HEIGHT
        self.reset_btn = Button(14, toolbar_y + 7, 100, 25, "Reset View", style="menu")
        self.mode_btn = Button(118, toolbar_y + 7, 100, 25, "Nonlinear", style="menu")

        right_x = WINDOW_WIDTH - 180
        self.left_arrow_btn = Button(right_x, toolbar_y + 7, 40, 25, "<", style="menu")
        self.page_label_rect = pygame.Rect(right_x + 50, toolbar_y + 7, 50, 25)
        self.right_arrow_btn = Button(
            right_x + 110, toolbar_y + 7, 40, 25, ">", style="menu"
        )

        self.all_buttons = [
            self.file_btn,
            self.pref_btn,
            self.reset_btn,
            self.mode_btn,
            self.left_arrow_btn,
            self.right_arrow_btn,
        ]

    def adjust_pref(self, key, direction):
        """Move a preference one step and clamp it to its allowed range."""
        spec = PREFERENCE[key]
        if "choices" in spec:
            choices = spec["choices"]
            current_index = choices.index(self.preferences[key])
            self.preferences[key] = choices[(current_index + direction) % len(choices)]
            return

        val = self.preferences[key] + direction * spec["step"]
        val = max(spec["min"], min(spec["max"], val))
        val = round(val) if spec["is_int"] else round(val, 3)
        self.preferences[key] = val
        if key == "grid_points":
            self._mark_dirty()

    def get_grid_step(self):
        """Choose a readable major-grid interval for the current zoom."""
        target_divisions = max(1, int(self.num_points - 1))
        rough_step = self.view_span / target_divisions

        # Snap the desired interval to a small set of visually useful steps.
        mag = 10.0 ** math.floor(math.log10(rough_step))
        rel_step = rough_step / mag

        if rel_step < 1.5:
            return 1.0 * mag
        elif rel_step < 2.75:
            return 2.0 * mag
        elif rel_step < 5.0:
            return 3.5 * mag
        elif rel_step < 8.0:
            return 6.5 * mag
        else:
            return 10.0 * mag

    def _update_bounds(self):
        """Derive visible x/y bounds from the center and square span."""
        self.x_bounds = (
            self.x_center - self.view_span / 2,
            self.x_center + self.view_span / 2,
        )
        self.y_bounds = (
            self.y_center - self.view_span / 2,
            self.y_center + self.view_span / 2,
        )

    def data_to_px(self, x_val, y_val):
        """Convert data coordinates to Pygame pixel coordinates."""
        x_range = self.x_bounds[1] - self.x_bounds[0]
        y_range = self.y_bounds[1] - self.y_bounds[0]
        px = (
            self.grid_area.left
            + (x_val - self.x_bounds[0]) / x_range * self.grid_area.width
        )
        # Pixel y increases downward, so data-space y must be inverted.
        py = (
            self.grid_area.top
            + (self.y_bounds[1] - y_val) / y_range * self.grid_area.height
        )
        return px, py

    def px_to_data(self, px, py):
        """Convert Pygame pixel coordinates back to data coordinates."""
        fx = (px - self.grid_area.left) / self.grid_area.width
        fy = (py - self.grid_area.top) / self.grid_area.height
        x_val = self.x_bounds[0] + fx * (self.x_bounds[1] - self.x_bounds[0])
        y_val = self.y_bounds[1] - fy * (self.y_bounds[1] - self.y_bounds[0])
        return x_val, y_val

    def zoom(self, amount, mouse_pos):
        """Zoom while keeping the point beneath the cursor stationary."""
        if amount == 0:
            return

        mx, my = self.px_to_data(*mouse_pos)
        speed = self.preferences["zoom_speed"]

        factor = 1.0 + (0.10 * speed * abs(amount))

        if amount > 0:
            new_span = self.view_span / factor
        else:
            new_span = self.view_span * factor

        if new_span < 0.001 or new_span > 100000:
            return

        self.view_span = new_span

        fx = (mouse_pos[0] - self.grid_area.left) / self.grid_area.width
        fy = (mouse_pos[1] - self.grid_area.top) / self.grid_area.height

        # Recenter so the cursor still corresponds to the original data point.
        self.x_center = mx + new_span / 2 - fx * new_span
        self.y_center = my - new_span / 2 + fy * new_span
        self._mark_dirty()

    def handle_events(self):
        """Process one frame of window, control, zoom, and pan events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            file_button_toggled = False
            pref_button_toggled = False
            any_button_consumed = False

            # Buttons get the event before it can start a canvas interaction.
            for button in self.all_buttons:
                if button.handle_event(event):
                    any_button_consumed = True
                    if button == self.file_btn:
                        if self.file_dropdown.visible:
                            self.file_dropdown.hide()
                        else:
                            self.file_dropdown.show()
                            self.preferences_panel.hide()
                        file_button_toggled = True
                    elif button == self.pref_btn:
                        if self.preferences_panel.visible:
                            self.preferences_panel.hide()
                        else:
                            self.preferences_panel.show()
                            self.file_dropdown.hide()
                        pref_button_toggled = True
                    elif button == self.reset_btn:
                        self.preferences["grid_points"] = 11
                        self.x_center = 5.0
                        self.y_center = 5.0
                        self.view_span = 10.0
                        self._mark_dirty()
                    elif button == self.mode_btn:
                        self.flow_type = (
                            "linear" if self.flow_type == "nonlinear" else "nonlinear"
                        )
                        self.mode_btn.text = (
                            "Linear" if self.flow_type == "linear" else "Nonlinear"
                        )
                        self._mark_dirty()
                    elif button == self.left_arrow_btn:
                        self.current_element_idx = max(0, self.current_element_idx - 1)
                        self._mark_dirty()
                    elif button == self.right_arrow_btn:
                        self.current_element_idx = min(
                            len(self.states_nxd) - 1,
                            self.current_element_idx + 1,
                        )
                        self._mark_dirty()

            if not file_button_toggled:
                self.file_dropdown.handle_event(event)
            if not pref_button_toggled:
                self.preferences_panel.handle_event(event)

            # Zoom only when a popup is not covering the canvas.
            if event.type == pygame.MOUSEWHEEL:
                if (
                    not self.file_dropdown.visible
                    and not self.preferences_panel.visible
                ):
                    mouse_pos = pygame.mouse.get_pos()
                    if self.grid_area.collidepoint(mouse_pos):
                        self.zoom(event.y, mouse_pos)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if (
                    not any_button_consumed
                    and not self.file_dropdown.visible
                    and not self.preferences_panel.visible
                    and self.grid_area.collidepoint(event.pos)
                ):
                    self.dragging = True
                    self.drag_last_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
                self.drag_last_pos = None

            elif event.type == pygame.MOUSEMOTION and self.dragging:
                # Convert pixel motion to a zoom-independent data-space offset.
                dx_px = event.pos[0] - self.drag_last_pos[0]
                dy_px = event.pos[1] - self.drag_last_pos[1]
                self.drag_last_pos = event.pos

                dx_data = (
                    -dx_px
                    * (self.view_span / self.grid_area.width)
                    * self.preferences["scroll_speed"]
                )
                dy_data = (
                    dy_px
                    * (self.view_span / self.grid_area.height)
                    * self.preferences["scroll_speed"]
                )

                self.x_center += dx_data
                self.y_center += dy_data
                self._mark_dirty()

        self._update_bounds()

    def _compute_flow_field(self):
        """Compute and cache vectors for the current view and data source."""
        # User-data mode evaluates around a selected trajectory sample; the
        # other branch evaluates the currently selected generated field.
        state_n = self.states_nxd[self.current_element_idx]
        inp_n = self.inputs_nxd[self.current_element_idx]
        delta_inp_n = None
        if self.delta_inputs_nxd is not None:
            delta_inp_n = self.delta_inputs_nxd[self.current_element_idx]

        flow = self._compute_user_flow(state_n, inp_n, delta_inp_n)

        self._flow_cache = {
            "grid": flow.grid.numpy(),
            "x_vel": flow.x_vels.numpy(),
            "y_vel": flow.y_vels.numpy(),
            "speed": flow.speeds.numpy(),
        }
        self._flow_dirty = False

    def _compute_user_flow(self, state_n, inp_n, delta_inp_n=None):
        """Calculate nonlinear or linear flow around one user sample."""
        finder = self.current_field()["finder"]

        num_points = self.preferences["grid_points"]
        x_offset = self.view_span / 2.0
        y_offset = self.view_span / 2.0

        finder.num_points = num_points
        finder.x_offset = x_offset
        finder.y_offset = y_offset
        finder.x_center = self._snap_to_grid(self.x_center)
        finder.y_center = self._snap_to_grid(self.y_center)

        bounds = (
            self.x_bounds[0],
            self.x_bounds[1],
            self.y_bounds[0],
            self.y_bounds[1],
        )

        if len(state_n.shape) == 1:
            state_n = state_n.unsqueeze(0)
        if len(inp_n.shape) == 1:
            inp_n = inp_n.unsqueeze(0)

        # Make a 2-D display grid, then lift it into RNN hidden-state space.
        low_dim_grid, inverse_grid = finder._inverse_grid(*bounds)

        if self.flow_type == "nonlinear":
            full_inp_batch = inp_n.repeat(low_dim_grid.shape[0], 1)
            with torch.no_grad():
                _, h = finder.rnn(
                    full_inp_batch.unsqueeze(finder.time_dim), inverse_grid.unsqueeze(0)
                )
        else:
            if delta_inp_n is None:
                raise ValueError("delta_inp required for linear flow")
            if len(delta_inp_n.shape) == 1:
                delta_inp_n = delta_inp_n.unsqueeze(0)
            delta_h = inverse_grid - state_n
            with torch.no_grad():
                h = finder.linearization(inp_n, state_n, delta_inp_n, delta_h)

        # Project predictions back to 2-D and derive vectors and speeds.
        h_next = finder._reduce_traj(h)
        x_vel, y_vel = finder._compute_velocity(h_next, low_dim_grid)
        speed = finder._compute_speed(x_vel, y_vel)
        x_vel, y_vel, low_dim_grid, speed = finder._reshape_vals(
            x_vel, y_vel, low_dim_grid, speed
        )

        return FlowField(x_vel, y_vel, low_dim_grid, speed)

    def draw_grid(self):
        """Draw grid lines, flow arrows, and arrowheads."""
        pygame.draw.rect(self.screen, WHITE, self.grid_area)

        # Prevent plot contents from spilling into the surrounding UI.
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(self.grid_area)

        step = self.get_grid_step()
        x_start_mult = math.floor(self.x_bounds[0] / step)
        x_end_mult = math.ceil(self.x_bounds[1] / step)
        y_start_mult = math.floor(self.y_bounds[0] / step)
        y_end_mult = math.ceil(self.y_bounds[1] / step)

        sub = self.small_grid
        sub_step = step / sub

        for mx in range(x_start_mult - 1, x_end_mult + 2):
            gx = mx * step
            px, _ = self.data_to_px(gx, 0)
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (px, self.grid_area.top),
                (px, self.grid_area.bottom),
                1,
            )
            if mx <= x_end_mult:
                for k in range(1, sub):
                    px_minor, _ = self.data_to_px(gx + k * sub_step, 0)
                    pygame.draw.line(
                        self.screen,
                        (240, 240, 240),
                        (px_minor, self.grid_area.top),
                        (px_minor, self.grid_area.bottom),
                        1,
                    )

        for my in range(y_start_mult - 1, y_end_mult + 2):
            gy = my * step
            _, py = self.data_to_px(0, gy)
            pygame.draw.line(
                self.screen,
                LIGHT_GRAY,
                (self.grid_area.left, py),
                (self.grid_area.right, py),
                1,
            )
            if my <= y_end_mult:
                for k in range(1, sub):
                    _, py_minor = self.data_to_px(0, gy + k * sub_step)
                    pygame.draw.line(
                        self.screen,
                        (240, 240, 240),
                        (self.grid_area.left, py_minor),
                        (self.grid_area.right, py_minor),
                        1,
                    )

        # Reuse the field until navigation or settings invalidate it.
        if self._flow_dirty or self._flow_cache is None:
            self._compute_flow_field()

        unit_px = (
            abs(
                self.data_to_px(self.x_bounds[0] + step, 0)[0]
                - self.data_to_px(self.x_bounds[0], 0)[0]
            )
            or 1
        )
        arrow_scale = unit_px * 0.35 * self.preferences["arrow_size_mult"]

        arrow_colormap = colormaps[self.preferences["colormap"]]

        cache = self._flow_cache
        grid = cache["grid"]
        x_vel = cache["x_vel"]
        y_vel = cache["y_vel"]
        speed = cache["speed"]

        # Normalize arrow lengths relative to the fastest visible vector.
        max_speed = float(np.max(speed)) if np.max(speed) > 0 else 1.0

        n_i, n_j = grid.shape[0], grid.shape[1]
        for i in range(n_i):
            for j in range(n_j):
                gx, gy = float(grid[i, j, 0]), float(grid[i, j, 1])
                px, py = self.data_to_px(gx, gy)

                if not (
                    -50 <= px - self.grid_area.left <= self.grid_area.width + 50
                    and -50 <= py - self.grid_area.top <= self.grid_area.height + 50
                ):
                    continue

                vx, vy = float(x_vel[i, j]), float(y_vel[i, j])
                mag = math.sqrt(vx * vx + vy * vy)
                s = float(speed[i, j])

                # Preserve direction while normalized speed controls length.
                if mag > 1e-9:
                    unit_dx, unit_dy = vx / mag, vy / mag
                else:
                    unit_dx, unit_dy = 0.0, 0.0

                length_frac = s / max_speed if max_speed > 0 else 0.0
                length_frac = max(0.0, min(1.0, length_frac))
                arrow_len = arrow_scale * (0.3 + 0.7 * length_frac)
                rgba = arrow_colormap(length_frac)
                arrow_color = tuple(round(channel * 255) for channel in rgba[:3])

                end_x = px + unit_dx * arrow_len
                end_y = py - unit_dy * arrow_len

                pygame.draw.line(self.screen, arrow_color, (px, py), (end_x, end_y), 2)

                if length_frac > 0.05:
                    angle = math.atan2(-unit_dy, unit_dx)
                    head_len = 8
                    p1 = (
                        end_x - head_len * math.cos(angle - math.pi / 6),
                        end_y - head_len * math.sin(angle - math.pi / 6),
                    )
                    p2 = (
                        end_x - head_len * math.cos(angle + math.pi / 6),
                        end_y - head_len * math.sin(angle + math.pi / 6),
                    )
                    pygame.draw.polygon(
                        self.screen, arrow_color, [p1, (end_x, end_y), p2]
                    )

        self.screen.set_clip(prev_clip)

    def _snap_to_grid(self, value):
        """Round a coordinate to the nearest major-grid line."""
        step = self.get_grid_step()
        return round(value / step) * step

    def draw_axes(self):
        """Draw border axes, tick marks, and numeric labels."""
        font = pygame.font.Font(None, 24)
        x_axis_y = self.grid_area.bottom
        y_axis_x = self.grid_area.left

        pygame.draw.line(
            self.screen,
            BLACK,
            (self.grid_area.left, x_axis_y),
            (self.grid_area.right, x_axis_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            BLACK,
            (y_axis_x, self.grid_area.top),
            (y_axis_x, self.grid_area.bottom),
            2,
        )

        step = self.get_grid_step()
        x_start_mult = math.floor(self.x_bounds[0] / step)
        x_end_mult = math.ceil(self.x_bounds[1] / step)
        y_start_mult = math.floor(self.y_bounds[0] / step)
        y_end_mult = math.ceil(self.y_bounds[1] / step)

        for mx in range(x_start_mult, x_end_mult + 1):
            gx = mx * step
            px, _ = self.data_to_px(gx, 0)
            if self.grid_area.left - 1 <= px <= self.grid_area.right + 1:
                pygame.draw.line(
                    self.screen, BLACK, (px, x_axis_y), (px, x_axis_y + 8), 2
                )
                label = font.render(fmt_num(gx), True, BLACK)
                self.screen.blit(label, label.get_rect(center=(px, x_axis_y + 20)))

        for my in range(y_start_mult, y_end_mult + 1):
            gy = my * step
            _, py = self.data_to_px(0, gy)
            if self.grid_area.top - 1 <= py <= self.grid_area.bottom + 1:
                pygame.draw.line(
                    self.screen, BLACK, (y_axis_x, py), (y_axis_x - 8, py), 2
                )
                label = font.render(fmt_num(gy), True, BLACK)
                self.screen.blit(label, label.get_rect(center=(y_axis_x - 30, py)))

    def draw_bounds_text(self):
        """Show the visible coordinate bounds in the top bar."""
        label_font = pygame.font.Font(None, 24)
        bounds_text = (
            f"x: ( {fmt_num(self.x_bounds[0])} , {fmt_num(self.x_bounds[1])} )      \
        y: ( {fmt_num(self.y_bounds[0])} , {fmt_num(self.y_bounds[1])} )"
        )
        surface = label_font.render(bounds_text, True, DARK_GRAY)
        x = WINDOW_WIDTH - surface.get_width() - 20
        y = (TOP_BAR_HEIGHT - surface.get_height()) // 2
        self.screen.blit(surface, (x, y))

    def draw_top_bar(self):
        """Draw the top bar and its menus over the plot."""
        top_bar_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, top_bar_rect)
        pygame.draw.line(
            self.screen,
            (218, 218, 223),
            (0, top_bar_rect.bottom),
            (WINDOW_WIDTH, top_bar_rect.bottom),
            1,
        )

        self.file_btn.is_active = self.file_dropdown.visible
        self.pref_btn.is_active = self.preferences_panel.visible
        self.file_btn.draw(self.screen)
        self.pref_btn.draw(self.screen)

        self.draw_bounds_text()

        self.file_dropdown.draw(self.screen)
        self.preferences_panel.draw(self.screen)

    def draw_toolbar(self):
        """Draw bottom controls and the page or sample indicator."""
        toolbar_rect = pygame.Rect(
            0, WINDOW_HEIGHT - TOOLBAR_HEIGHT, WINDOW_WIDTH, TOOLBAR_HEIGHT
        )
        pygame.draw.rect(self.screen, WHITE, toolbar_rect)
        pygame.draw.line(
            self.screen,
            (218, 218, 223),
            (0, toolbar_rect.top),
            (WINDOW_WIDTH, toolbar_rect.top),
            1,
        )

        self.reset_btn.draw(self.screen)
        self.mode_btn.draw(self.screen)
        self.left_arrow_btn.draw(self.screen)
        self.right_arrow_btn.draw(self.screen)

        page_font = pygame.font.SysFont("Arial", 14)
        page_text = page_font.render(
            f"{self.current_element_idx + 1}/{len(self.states_nxd)}",
            True,
            BLACK,
        )
        self.screen.blit(
            page_text, page_text.get_rect(center=self.page_label_rect.center)
        )

    def run(self):
        """Run the 60-FPS event/update/draw loop until the window closes."""
        while self.running:
            self.handle_events()
            self.screen.fill(CANVAS_BG)

            self.draw_grid()
            self.draw_axes()
            self.draw_top_bar()
            self.draw_toolbar()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()
