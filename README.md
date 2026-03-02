# EMPulse-1D — FDTD Source Localization (Inverse Problem)

A minimal, **1D FDTD** project focused on the **inverse problem**:

> **Estimate the location** of a point-like dipole source along a 1D domain \(z \in [0, L]\)  
> given **electric field measurements** \(E(t)\) at **both boundaries**.

This repo intentionally keeps the physics and codebase small so the **inverse pipeline** (simulation → boundary traces → localization → plots) is rock-solid before expanding to 2D.

---

## Project Goal

### Forward problem
Given:
- 1D domain along \(z\)
- a **point source** at position \(z_0\)
- a known **Ricker wavelet** time dependence
- unknown **source amplitude** \(A\)
- **Mur absorbing boundaries** at \(z=0\) and \(z=L\)

simulate the fields and record:
- \(E(t)\) at the **left boundary**
- \(E(t)\) at the **right boundary**

### Inverse problem (localization)
Given measured boundary traces:
\[
y(t) = \{E_L(t), E_R(t)\}
\]
estimate the source location:
\[
\hat{z}_0 = \arg\min_{z_0 \in [0,L]} \; \min_{A} \; \mathcal{L}\Big(y(t),\, A \,\hat{y}(t; z_0)\Big)
\]

We will:
- perform a **grid search over candidate** \(z_0\) positions (on the FDTD grid)
- **fit amplitude \(A\)** in closed form for each candidate
- choose the \(z_0\) with the lowest loss
- visualize the loss curve and trace fits

---

## Model: 1D Yee FDTD (Plane-Wave / Transverse Fields)

We use a standard **1D Yee grid** (propagation along \(z\)) with transverse fields:
- \(E_x(z,t)\)
- \(H_y(z,t)\)

This is a canonical 1D Maxwell update scheme.  
The “dipole” is represented as a **localized impressed source** injected at one grid cell.

> Note: A true 3D dipole is not representable in 1D, but this formulation is ideal for building and testing the inverse localization pipeline.

---

## Source Waveform: Ricker Wavelet

We use a band-limited **Ricker wavelet** (Mexican hat) as the time dependence.

Typical form:
\[
s(t)=\Big(1 - 2\pi^2 f_0^2 (t-t_0)^2\Big)\exp\Big(-\pi^2 f_0^2 (t-t_0)^2\Big)
\]

Parameters:
- center frequency \(f_0\)
- time shift \(t_0\) (so the wavelet starts near-zero)

Amplitude \(A\) is treated as **unknown** and fit during inversion.

---

## Boundaries: Mur Absorbing Boundary Condition

To suppress reflections (critical for inversion stability), we use **1st-order Mur ABC** at both ends.

Mur is lightweight and works well for the initial MVP.  
(We can replace/extend with PML in later phases.)



