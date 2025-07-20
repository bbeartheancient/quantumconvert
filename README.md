# quantum to classical converter
requirements:
- Python 3.12
- numpy
- scipy
- qiskit
- qiskit_aer
- matplotlib
- astropy
# qconvertxfits.py:
best version, allows for manual H0 value input or FITS file parsing
//don't forget to correct the file paths as i was lazy and didn't remove them from the script
# qconvert.py:
measures quantum values between 3 qubits for transferring wave particles.
//sums quantum vectors before state transmission, so a large number of wave particles can be simulated.
# qconvertH01.py:
compares different Hubble constants through dynamically assigned qubits set by N.  #WARNING###: requires a lot of RAM with large N values, keep values small unless you have a lot of RAM

# ABStract:
The interplay between quantum mechanics and cosmology has garnered significant in-
terest, particularly in exploring how quantum correlations might manifest in macro-
scopic, astrophysical phenomena. This script introduces a quantum-classical conver-
sion calculator implemented in Python using Qiskit, which computes quantum corre-
lation parameters for an n-qubit GHZ state and contrasts them with classical analogs.
Additionally, the model integrates cosmological parameters, such as redshift and dark
matter density, to study their influence on quantum states via density matrix scaling.
The quantum framework is based on ambisonic theory (Gerzon, 1975). Ambison-
ics is a surround sound format that creates a spherical wavestate from its individual
sources by routing the signals directionally through a Hadamard matrix. This forms a
local quantum state comprised of sound waves, but the routing is nearly identical for
matter.
The calculator serves as a tool to bridge quantum information theory with cosmo-
logical dynamics, providing a framework to quantify correlations and explore their be-
havior under cosmological evolution.

The calculator incorporates a flat ΛCDM cosmological model, defined by the density pa-
rameters for radiation (Ωr), matter (Ωm), dark energy (ΩΛ), and curvature (Ωk), with val-
1
ues:
Ωr = 9.05 × 10−5, Ωm = 0.3, ΩΛ = 0.7, Ωk = 1 − Ωm − ΩΛ − Ωr.
The Hubble parameter at scale factor a is given by:
H(a) = H0
√ Ωr
a2 + Ωm
a + Ωk + ΩΛa2,
where H0 = 2.268 × 10−18 s−1 is the Hubble constant. The age of the universe at scale
factor a is computed via:
t(a) = 1
H0
∫ a
0
da′
√ Ωr
a′2 + Ωm
a′ + Ωk + ΩΛa′2
.
The redshift z is related to the scale factor by z = 1
a − 1. For a given lookback time t (time
ago from the present age t0 = 13.8 × 109 years), the redshift is computed by solving:
teffective = t0 − t = 1
H0
∫ a
0
da′
√ Ωr
a′2 + Ωm
a′ + Ωk + ΩΛa′2
,
using numerical root-finding to determine a.
The effective speed of light ceff(t) accounts for a hypothetical decay, modeled as:
ceff(t) = c
(
1 − δe−t/τ )
,
where c = 3.00×108 m/s, δ = 0.1, and τ = 1.0×109 years. The dark matter number density
NDM is scaled by the effective speed of light and redshift:
NDM(t, z) = N0
ceff(t0)
ceff(t) (1 + z)3,
where N0 = 3.0 × 106 m−3 is the present dark matter density.
2.2 Quantum Correlations
The quantum correlations are computed for an n-qubit GHZ state, defined as:
|ψ⟩ = |0⟩⊗n + |1⟩⊗n
√2 .
The quantum circuit is constructed with a Hadamard gate on the first qubit followed
by CNOT gates to entangle subsequent qubits. The expectation value of the operator
Z1Z2 · · · Zn is calculated from measurement outcomes in the computational basis:
⟨Z1Z2 · · · Zn⟩ = P (|0⟩⊗n) + P (|1⟩⊗n) − ∑
other states
P (state),
where P (state) is the probability of measuring a given state, obtained from 105 shots on
the Qiskit AerSimulator. The quantum correlation parameter Squantum is:
Squantum = 2√2√N ⟨Z1Z2 · · · Zn⟩,
where N is the number of particles. The parameter W is:
W = Squantum
√2.
The parameter Mquantum depends on the parity of N :
Mquantum =
{
W √N /√2 if N is even,
W √N /√N + 1 if N is odd.
2
2.3 Classical Correlations
Classical correlations are computed as simpler analogs to the quantum parameters:
Sclassical = 2√N , Wclassical = Sclassical
√2 , Mclassical = Wclassical.
These assume a classical system with no entanglement, providing a baseline for com-
parison.
2.4 Density Matrix and Cosmological Scaling
The density matrix ρ for the GHZ state is computed from the statevector:
ρ = |ψ⟩⟨ψ|.
When a redshift z is provided, the density matrix is scaled by (1 + z)3 to reflect cosmo-
logical expansion effects, and normalized to ensure Tr(ρ) = 1:
ρ′ = ρ(1 + z)3
Tr(ρ(1 + z)3) .
The purity of the density matrix is:
γ = Tr(ρ2).
The expectation value of Z1Z2 · · · Zn is also computed using the projector:
P = |0⟩⊗n⟨0|⊗n + |1⟩⊗n⟨1|⊗n,
⟨Z1Z2 · · · Zn⟩ = ⟨ψ|P |ψ⟩.
3 Implementation
The calculator is implemented in Python using Qiskit for quantum simulations and SciPy
for numerical integration and root-finding. The main components are:
• Redshift Calculation: Computes z for a given lookback time t by solving for the
scale factor a using Brent’s method.
• Quantum Simulation: Constructs an n-qubit GHZ state, simulates measurements,
and computes Squantum, Mquantum, and the density matrix.
• Classical Calculation: Computes Sclassical and Mclassical for comparison.
• Cosmological Parameters: Calculates ceff and NDM based on t and z.
The code includes debugging statements to verify intermediate values and conver-
gence, ensuring numerical stability. A histogram of measurement outcomes is saved for
visualization.
3
4 Quantum-Classical Correlations
The quantum and classical correlation parameters exhibit distinct behaviors. For a GHZ
state, ⟨Z1Z2 · · · Zn⟩ ≈ 1 ideally, leading to:
Squantum ≈ 2√2√N , Sclassical = 2√N .
Thus, Squantum exceeds Sclassical by a factor of √2, reflecting the enhanced correlations
due to quantum entanglement. The parameter Mquantum introduces an N -dependent
scaling that modulates the quantum advantage, particularly for odd N , where the de-
nominator √N + 1 slightly reduces the magnitude compared to even N .
The density matrix scaling by (1 + z)3 models the effect of cosmological expansion on
quantum states, analogous to the scaling of matter density. The purity γ ≈ 1 for the GHZ
state confirms its coherence, while deviations in ⟨Z1Z2 · · · Zn⟩ between measurement and
statevector methods highlight numerical precision limits in simulations.
5 Results and Discussion
For N = 10, nqubits = 3, and a lookback time t = 1.0 × 109 years, the calculator yields:
• Redshift z ≈ 0.087, computed numerically.
• Squantum ≈ 8.944, Mquantum ≈ 11.255 (odd N ), compared to Sclassical = 6.325, Mclassical =
4.472.
• ceff ≈ 2.973 × 108 m/s, NDM ≈ 3.773 × 106 m−3.
• Density matrix purity γ ≈ 1.0, with ⟨Z1Z2 · · · Zn⟩ ≈ 0.999.
The quantum correlations consistently exceed classical ones, demonstrating the im-
pact of entanglement. The cosmological scaling of ρ suggests a framework for studying
quantum states in an expanding universe, potentially relevant to quantum cosmology
models. Future work could extend the model to include decoherence effects or alterna-
tive quantum states.
6 Conclusion
The quantum-classical conversion calculator provides a novel approach to studying quan-
tum correlations in a cosmological context. By integrating Qiskit simulations with cos-
mological parameter calculations, it offers insights into the interplay between quantum
entanglement and macroscopic phenomena. The derived formulas and their computa-
tional implementation enable quantitative comparisons, highlighting the quantum ad-
vantage in correlation strength and the influence of cosmological expansion on quan-
tum states. By linking the Z vector of the quantum state to the temporal reaction of
dark matter decoherence, we restore the negative phase of the timestream that may be
phased outside of our perception along this vector.
