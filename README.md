# EMPulseModel  
**Time-Reversal Electromagnetic Pulse Focusing via FDTD**

EMPulseModel is a **computational electromagnetics project** that implements a **time-reversal electromagnetic (TR-EM) framework** using **finite-difference time-domain (FDTD)** simulations. The project studies EM wave propagation, sensor-based field recording, and time-reversed re-propagation leading to spatial refocusing at the original source location.

---

## Overview

Maxwell’s equations in linear, lossless media are **time-reversal invariant**. As a result, electromagnetic waves recorded at a set of sensors can be reversed in time and re-emitted into the medium, causing energy to refocus at the original source location.

This project demonstrates that principle numerically by:

1. Simulating forward EM pulse propagation using FDTD  
2. Recording fields at discrete sensor locations  
3. Time-reversing the recorded signals  
4. Re-propagating the reversed fields  
5. Observing refocusing at the source position  

---

## Physical Model

The simulations are based on Maxwell’s equations in the time domain:

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t},
\qquad
\nabla \times \mathbf{H} = \frac{\partial \mathbf{D}}{\partial t} + \mathbf{J}
$$

with constitutive relations:

$$
\mathbf{D} = \epsilon \mathbf{E}, \qquad
\mathbf{B} = \mu \mathbf{H}
$$

In homogeneous, lossless media, these equations admit wave solutions that are invariant under time reversal.

---

## Numerical Method

### Finite-Difference Time-Domain (FDTD)

- Spatial discretization using a **Yee grid**
- Staggered electric and magnetic field components
- Explicit time stepping
- Stability enforced by the **Courant–Friedrichs–Lewy (CFL)** condition

### Boundary Treatment

- Absorbing boundary conditions to minimize artificial reflections
- Grid resolution chosen relative to wavelength to control numerical dispersion

---

## Time-Reversal Methodology

### Forward Propagation
- A broadband EM pulse is emitted from a localized source
- Fields propagate through the computational domain
- Field values are recorded at sensor locations

### Time Reversal
- Recorded signals are reversed in time
- Time-reversed signals are re-introduced at the sensor locations
- Fields propagate backward through the domain

### Refocusing
- Energy refocuses at the original source location
- Refocusing quality depends on:
  - Bandwidth
  - Sensor coverage
  - Numerical dispersion
  - Boundary reflections

---

## Future Work

- Introduce ML Assistance  to accelerate the model.

---

## Author

Pedro Martins



