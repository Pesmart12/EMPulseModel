# Time-Reversal Electromagnetic Pulse Focusing (FDTD Simulation)

This project implements a 2D Finite-Difference Time-Domain (FDTD) simulation of electromagnetic wave propagation in the TMᶻ (oscilates perpendicular to the z direction in which it propogates) polarization, demonstrating time-reversal focusing—a phenomenon where EM waves recorded at a boundary can be played back in reverse to reconstruct and refocus the original pulse at its source.

Time reversal: the wave equation (for acoustics, electromagnetism, quantum wavefunctions) is time-symmetric --> If you record a wave leaving a source and then play the recording backward into the system, the wave will focus precisely back at the original emission point. This project simulates that process:

  1. Forward Run
     - Launch a short EM pulse from an interior source.
     - Record the electric field E_z at boundary “detector” locations for all times.

  2. Time-Reversed Run
     - Reset the fields.
     - Replay the recorded boundary fields in reverse time order.
     - Observe the wave refocusing at the original source point.

This replicates the operation of a time-reversal mirror for electromagnetic waves.
