"""
infection_immune_abm.py

Agent-based model of bacterial infection and neutrophil immune response
using the Mesa framework. Runs a 2D simulation and saves an animated GIF
called "infection_immune_response.gif".
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation


class Bacterium(Agent):
    """
    Bacterial agent that can proliferate near the infection center.
    """

    def __init__(self, unique_id, model, p_divide=0.25):
        super().__init__(unique_id, model)
        self.p_divide = p_divide

    def step(self):
        # Stochastic division into a neighboring cell if space is available
        if self.random.random() < self.p_divide:
            neighbors = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            candidate_positions = []
            for pos in neighbors:
                cellmates = self.model.grid.get_cell_list_contents([pos])
                # Only divide into locations without another bacterium
                if not any(isinstance(a, Bacterium) for a in cellmates):
                    candidate_positions.append(pos)

            if candidate_positions:
                new_pos = self.random.choice(candidate_positions)
                new_bacterium = Bacterium(
                    self.model.next_id(), self.model, p_divide=self.p_divide
                )
                self.model.grid.place_agent(new_bacterium, new_pos)
                self.model.schedule.add(new_bacterium)


class Neutrophil(Agent):
    """
    Neutrophil agent that performs chemotaxis toward bacteria
    and kills them on contact.
    """

    def step(self):
        # First, attempt to kill bacteria in the current cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        bacteria_here = [a for a in cellmates if isinstance(a, Bacterium)]
        if bacteria_here:
            target = self.random.choice(bacteria_here)
            self.model.grid.remove_agent(target)
            self.model.schedule.remove(target)
            return

        # If no bacteria here, move using a simple chemotaxis rule
        bacteria_positions = [
            a.pos for a in self.model.schedule.agents if isinstance(a, Bacterium)
        ]
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True
        )

        if bacteria_positions:
            # Chemotaxis: move to the neighbor that minimizes
            # distance to the nearest bacterium
            def distance_to_closest_bacterium(pos):
                x, y = pos
                return min((x - bx) ** 2 + (y - by) ** 2 for bx, by in bacteria_positions)

            best_dist = None
            best_cells = []
            for pos in neighbors:
                d = distance_to_closest_bacterium(pos)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_cells = [pos]
                elif d == best_dist:
                    best_cells.append(pos)
            new_pos = self.random.choice(best_cells)
        else:
            # No bacteria present: random walk
            new_pos = self.random.choice(neighbors)

        self.model.grid.move_agent(self, new_pos)


class InfectionImmuneModel(Model):
    """
    2D tissue model with bacteria in the center and neutrophils
    invading from the tissue boundaries.
    """

    def __init__(
        self,
        width=50,
        height=50,
        initial_bacteria=8,
        n_neutrophils=120,
        p_divide=0.22,
        max_steps=120,
        output_dir="frames",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.initial_bacteria = initial_bacteria
        self.n_neutrophils = n_neutrophils
        self.p_divide = p_divide
        self.max_steps = max_steps
        self.output_dir = output_dir

        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        self._prepare_output_dir()
        self._create_initial_bacteria()
        self._create_neutrophils_at_boundaries()

    # --- Initialization helpers -------------------------------------------------

    def _prepare_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for f in glob.glob(os.path.join(self.output_dir, "frame_*.png")):
            try:
                os.remove(f)
            except OSError:
                pass

    def _create_initial_bacteria(self):
        cx, cy = self.width // 2, self.height // 2
        radius = 2
        placed = 0
        attempts = 0
        max_attempts = 200

        while placed < self.initial_bacteria and attempts < max_attempts:
            attempts += 1
            dx = self.random.randint(-radius, radius)
            dy = self.random.randint(-radius, radius)
            x = min(max(cx + dx, 0), self.width - 1)
            y = min(max(cy + dy, 0), self.height - 1)
            pos = (x, y)
            cellmates = self.grid.get_cell_list_contents([pos])
            if any(isinstance(a, Bacterium) for a in cellmates):
                continue

            bacterium = Bacterium(self.next_id(), self, p_divide=self.p_divide)
            self.grid.place_agent(bacterium, pos)
            self.schedule.add(bacterium)
            placed += 1

    def _create_neutrophils_at_boundaries(self):
        # Create neutrophils uniformly along all four edges
        possible_positions = []

        # Top and bottom edges
        for x in range(self.width):
            possible_positions.append((x, 0))
            possible_positions.append((x, self.height - 1))

        # Left and right edges
        for y in range(1, self.height - 1):
            possible_positions.append((0, y))
            possible_positions.append((self.width - 1, y))

        self.random.shuffle(possible_positions)
        for i in range(min(self.n_neutrophils, len(possible_positions))):
            pos = possible_positions[i]
            neut = Neutrophil(self.next_id(), self)
            self.grid.place_agent(neut, pos)
            self.schedule.add(neut)

    # --- Simulation loop --------------------------------------------------------

    def step(self):
        self.schedule.step()

    def run_model(self):
        for step in range(self.max_steps):
            self.step()
            self._save_frame(step)

    # --- Visualization ----------------------------------------------------------

    def _save_frame(self, step):
        # RGB image: base tissue background
        h, w = self.height, self.width
        tissue_color = np.array([1.0, 0.96, 0.92])  # soft beige
        img = np.ones((h, w, 3)) * tissue_color

        bacteria_map = np.zeros((h, w))
        neutrophil_map = np.zeros((h, w))

        for agent in self.schedule.agents:
            x, y = agent.pos
            if isinstance(agent, Bacterium):
                bacteria_map[y, x] += 1
            elif isinstance(agent, Neutrophil):
                neutrophil_map[y, x] += 1

        # Normalize maps for visualization
        if bacteria_map.max() > 0:
            bacteria_norm = bacteria_map / bacteria_map.max()
        else:
            bacteria_norm = bacteria_map

        if neutrophil_map.max() > 0:
            neutrophil_norm = neutrophil_map / neutrophil_map.max()
        else:
            neutrophil_norm = neutrophil_map

        # Overlay bacteria (red) and neutrophils (blue)
        # img[..., 0] = red channel, img[..., 2] = blue channel
        img[..., 0] += 0.6 * bacteria_norm  # more red where bacteria cluster
        img[..., 2] += 0.7 * neutrophil_norm  # more blue where neutrophils cluster

        # Clip to valid range
        img = np.clip(img, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, interpolation="nearest", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Infection and neutrophil response – step {step}", fontsize=10)
        plt.tight_layout()

        frame_path = os.path.join(self.output_dir, f"frame_{step:03d}.png")
        fig.savefig(frame_path, dpi=150)
        plt.close(fig)

    # --- GIF export -------------------------------------------------------------

    def make_gif(self, gif_name="infection_immune_response.gif", duration=0.15):
        frame_files = sorted(
            glob.glob(os.path.join(self.output_dir, "frame_*.png"))
        )
        if not frame_files:
            raise RuntimeError("No frames found to build GIF.")

        frames = [imageio.imread(f) for f in frame_files]
        imageio.mimsave(gif_name, frames, duration=duration)


def main():
    model = InfectionImmuneModel(
        width=50,
        height=50,
        initial_bacteria=10,
        n_neutrophils=140,
        p_divide=0.20,
        max_steps=120,
        output_dir="frames",
    )
    model.run_model()
    model.make_gif("infection_immune_response.gif")
    print('Animated GIF saved as "infection_immune_response.gif"')


if __name__ == "__main__":
    main()
