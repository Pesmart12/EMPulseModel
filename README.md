# Time-Reversal Electromagnetic Pulse Focusing (FDTD Simulation)

This project implements a 2D Finite-Difference Time-Domain (FDTD) simulation of electromagnetic wave propagation in the TMᶻ (oscilates perpendicular to the z direction in which it propogates) polarization, demonstrating time-reversal focusing—a phenomenon where EM waves recorded at a boundary can be played back in reverse to reconstruct and refocus the original pulse at its source.

Time reversal: the wave equation (for acoustics, electromagnetism, quantum wavefunctions) is time-symmetric --> If you record a wave leaving a source and then play the recording backward into the system, the wave will focus precisely back at the original emission point. This project simulates that process:

---

## Physical Model

The simulations are based on **Maxwell’s equations** in the time domain:

\[
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad
\nabla \times \mathbf{H} = \frac{\partial \mathbf{D}}{\partial t} + \mathbf{J}
\]

with constitutive relations:
\[
\mathbf{D} = \epsilon \mathbf{E}, \quad \mathbf{B} = \mu \mathbf{H}
\]

The system is discretized using a **Yee grid**, enabling stable and explicit time stepping of EM fields.

---

## Methodology

### 1. Forward Simulation
- Emit a broadband EM pulse from a source
- Propagate fields through a 2D or 3D domain
- Record field values at discrete sensor locations

### 2. Time-Reversal
- Reverse recorded signals in time
- Re-inject them into the simulation domain
- Observe spatial and temporal refocusing at the source

### 3. Inverse Modeling (ML-Assisted)
- Use recorded fields as input data
- Train a PyTorch model to:
  - Predict source location
  - Refine time-reversal reconstructions
- Enables robustness to noise and sparse sensor layouts

---

## Features

- FDTD solver using Yee discretization
- Time-reversal electromagnetic propagation
- Modular forward / inverse design
- PyTorch-based inverse source localization
- Extensible to:
  - Multiple sources
  - Noisy environments
  - Dispersive or inhomogeneous media

---

## Author
Pedro Martins


