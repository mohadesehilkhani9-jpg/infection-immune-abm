# Agent-Based Simulation of Infection and Neutrophil Response in Tissue

This repository contains a 2D agent-based model (ABM) of an infection focus and the surrounding innate immune response. Bacteria proliferate in the center of a tissue, while neutrophils enter from the boundaries, perform chemotaxis toward the infection, and kill bacteria upon contact. The model is implemented in Python using the [Mesa](https://mesa.readthedocs.io/) framework.

## Animated preview

The simulation automatically generates an animated GIF after each run:

![Infection and immune response](infection_immune_response.gif)

Bacteria (red) expand from a central focus, while neutrophils (blue) invade from the tissue edges, migrate up an effective chemotactic gradient, and clear the infection.

## Installation

Tested with Python 3.9+.

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate           # Windows
