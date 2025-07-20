import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import DensityMatrix, Statevector
import matplotlib.pyplot as plt
import logging
from astropy.io import fits

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
c = 3.00e8          # Speed of light in m/s
delta = 0.1         # Fractional reduction
tau = 1.0e9         # Decay timescale in years
t0 = 13.8e9         # Present age of the universe in years
c_eff_t0 = c * (1 - delta * np.exp(-t0 / tau))  # Precomputed c_eff at t0
N0 = 3.0e6          # Current dark matter density in m^-3
H0_default = 68  # Hubble constant in s^-1 (70 km/s/Mpc)
Omega_m_default = 0.3   # Matter density parameter
Omega_Lambda_default = 0.7  # Dark energy density parameter
Omega_r_default = 9.05e-5   # Radiation density parameter
Omega_k_default = 1 - Omega_m_default - Omega_Lambda_default - Omega_r_default
SECONDS_PER_YEAR = 3.15576e7  # Seconds in a year
KM_PER_MPC = 3.0857e19  # Kilometers per megaparsec
MPC_IN_M = 3.0857e22    # Meters per megaparsec
H0_CONVERSION = 1000 / MPC_IN_M  # Conversion from km/s/Mpc to s^-1
MAX_QUBITS = 8  # Maximum qubits for simulation
MAX_N = 1e6     # Maximum N to prevent overflow

# Debug constants
logger.info(f"SECONDS_PER_YEAR={SECONDS_PER_YEAR:.2e}")
logger.info(f"Precomputed c_eff_t0={c_eff_t0:.4e} m/s")
logger.info(f"Omega_r + Omega_m + Omega_Lambda + Omega_k = {Omega_r_default + Omega_m_default + Omega_Lambda_default + Omega_k_default:.6f}")

def hubble_integral(a, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Integrand for the age of the universe."""
    if a < 1e-20:
        a = 1e-20
    term = Omega_r / a**2 + Omega_m / a + Omega_k + Omega_Lambda * a**2
    if term <= 0:
        logger.warning(f"Negative or zero integrand at a={a:.6e}, setting to 1e-20")
        term = 1e-20
    result = 1 / np.sqrt(term)
    return result

def compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Compute age of the universe at scale factor a in seconds."""
    if a <= 0:
        logger.error(f"Invalid scale factor a={a:.6e}")
        return 0
    try:
        if a < 1e-6:
            integral1, err1 = quad(
                hubble_integral, 0, 1e-6,
                args=(Omega_r, Omega_m, Omega_k, Omega_Lambda),
                epsabs=1e-14, epsrel=1e-14, limit=200
            )
            integral2, err2 = quad(
                hubble_integral, 1e-6, a,
                args=(Omega_r, Omega_m, Omega_k, Omega_Lambda),
                epsabs=1e-14, epsrel=1e-14, limit=200
            )
            integral = integral1 + integral2
            err = err1 + err2
        else:
            integral, err = quad(
                hubble_integral, 0, a,
                args=(Omega_r, Omega_m, Omega_k, Omega_Lambda),
                epsabs=1e-14, epsrel=1e-14, limit=400
            )
        
        if err > 1e-8:
            logger.warning(f"High integration error at a={a:.6e}, err={err:.2e}")
        age = integral / H0
        return age
    except Exception as e:
        logger.error(f"Integration failed for a={a:.6e}: {e}")
        return 0

def compute_redshift(t, lookback=True, H0_value=H0_default, Omega_r=Omega_r_default, Omega_m=Omega_m_default, Omega_k=None, Omega_Lambda=Omega_Lambda_default):
    """Compute redshift z based on time t (in years)."""
    if t < 0:
        raise ValueError("t must be >= 0")
    
    Omega_k = 1 - Omega_m - Omega_Lambda - Omega_r if Omega_k is None else Omega_k
    t_effective = t0 - t if lookback else t
    if t_effective < 0:
        raise ValueError("Lookback time t cannot exceed present age t0")
    
    t_seconds = t_effective * SECONDS_PER_YEAR
    if abs(t_seconds / t_effective - SECONDS_PER_YEAR) > 1e-6:
        raise ValueError(f"Time conversion error: t_seconds={t_seconds:.2e}")
    
    if lookback and t / t0 < 0.01:
        z_approx = H0_value * t * SECONDS_PER_YEAR
        logger.info(f"Using Hubble's law approximation, z_approx={z_approx:.2e}")
        return z_approx
    
    def age_diff(a):
        return compute_age(a, H0_value, Omega_r, Omega_m, Omega_k, Omega_Lambda) - t_seconds
    
    a_values = np.logspace(-12, 0, 300)
    age_diff_values = [age_diff(a) for a in a_values]
    sign_changes = [(a_values[i], a_values[i+1]) for i in range(len(age_diff_values)-1) if age_diff_values[i] * age_diff_values[i+1] < 0]
    
    if not sign_changes:
        logger.error(f"No sign change found for t={t:.2e} years")
        z_approx = H0_value * t * SECONDS_PER_YEAR
        logger.warning(f"Using approximate redshift z={z_approx:.6f}")
        return z_approx
    
    bracket = sign_changes[0]
    logger.info(f"Using bracket [{bracket[0]:.2e}, {bracket[1]:.2e}] for t={t:.2e} years")
    
    try:
        result = root_scalar(age_diff, bracket=bracket, method='brentq', xtol=1e-14, rtol=1e-14)
        a = result.root
        z = 1 / a - 1
        if z > 1000:
            logger.warning(f"High redshift z={z:.2e} for t={t:.2e} years")
        logger.info(f"z={z:.6e}, converged={result.converged}")
        return max(z, 0)
    except ValueError as e:
        logger.error(f"Root finding failed: {e}")
        z_approx = H0_value * t * SECONDS_PER_YEAR
        logger.warning(f"Using approximate redshift z={z_approx:.6f}")
        return z_approx

def compute_quantum_qiskit(N, n_qubits=3, shots=100000):
    """Compute quantum correlations using Qiskit for an n-qubit GHZ state."""
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if n_qubits < 1 or n_qubits > MAX_QUBITS:
        raise ValueError(f"n_qubits must be between 1 and {MAX_QUBITS}")
    
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure(range(n_qubits), range(n_qubits))
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    expectation = 0
    for state, count in counts.items():
        if state in ['0' * n_qubits, '1' * n_qubits]:
            expectation += count / shots
        else:
            expectation -= count / shots
    
    expectation = max(min(expectation, 1.0), -1.0)
    
    S_quantum = 2 * np.sqrt(2) * np.sqrt(N) * expectation
    W = S_quantum * np.sqrt(2)
    M_quantum = W * np.sqrt(N) / np.sqrt(2) if N % 2 == 0 else W * np.sqrt(N) / np.sqrt(N + 1)
    
    plot_histogram(counts)
    plt.savefig(f'histogram_N{N}_nqubits{n_qubits}.png')
    plt.close()
    
    logger.info(f"Quantum S_quantum={S_quantum:.4f}, M_quantum={M_quantum:.4f}")
    return S_quantum, M_quantum, circuit, counts, expectation

def compute_density_matrix(circuit, z=None):
    """Compute density matrix for the circuit."""
    n_qubits = circuit.num_qubits
    circuit_without_measure = QuantumCircuit(n_qubits)
    circuit_without_measure.h(0)
    for i in range(n_qubits - 1):
        circuit_without_measure.cx(i, i + 1)
    
    simulator = AerSimulator()
    circuit_without_measure.save_statevector()
    job = simulator.run(circuit_without_measure, method='statevector')
    result = job.result()
    statevector = result.get_statevector(circuit_without_measure)
    
    logger.info(f"Statevector = {statevector}")
    
    rho = DensityMatrix(statevector)
    
    if z is not None:
        scale_factor = (1 + z)**3
        rho_data = rho.data * scale_factor
        trace = np.trace(rho_data)
        if abs(trace) > 1e-10:
            rho_data = rho_data / trace
        rho = DensityMatrix(rho_data)
    
    purity = np.real(np.trace(rho.data @ rho.data))
    
    state = Statevector(statevector)
    expectation_Z = sum(abs(state.data[int(state_str, 2)])**2 for state_str in ['0' * n_qubits, '1' * n_qubits])
    
    logger.info(f"GHZ correlation expectation = {expectation_Z:.6f}")
    return rho, purity, expectation_Z

def compute_classical(N):
    """Compute classical correlation parameters."""
    S_classical = 2 * np.sqrt(N)
    W_classical = S_classical / np.sqrt(2)
    M_classical = W_classical
    logger.info(f"Classical S_classical={S_classical:.4f}, M_classical={M_classical:.4f}")
    return S_classical, M_classical

def compute_cosmological(t, z):
    """Compute cosmological parameters."""
    if t < 0 or z < 0:
        raise ValueError("t and z must be >= 0")
    
    c_eff_t = c * (1 - delta * np.exp(-t / tau))
    N_DM = N0 * (c_eff_t0 / c_eff_t) * (1 + z)**3
    logger.info(f"c_eff_t={c_eff_t:.4e}, N_DM={N_DM:.4e}")
    return c_eff_t, N_DM

def validate_fits_file(fits_file):
    """Validate FITS file and extract parameters."""
    try:
        with fits.open(fits_file) as hdul:
            if not hdul:
                raise ValueError("Empty FITS file")
            header = hdul[0].header
            H0_km_s_Mpc = header.get('H0', 70)
            Omega_m = header.get('OMEGAM', 0.3)
            Omega_Lambda = header.get('OMEGAL', 0.7)
            Omega_r = header.get('OMEGAR', 9.05e-5)
            Omega_k = header.get('OMEGAK', 1 - Omega_m - Omega_Lambda - Omega_r)
            if not (0 < Omega_m < 1):
                raise ValueError(f"Invalid Omega_m={Omega_m}")
            if not (0 < Omega_Lambda < 1):
                raise ValueError(f"Invalid Omega_Lambda={Omega_Lambda}")
            if not (0 <= Omega_r < 1):
                raise ValueError(f"Invalid Omega_r={Omega_r}")
            if not (abs(Omega_k) < 1):
                raise ValueError(f"Invalid Omega_k={Omega_k}")
            if H0_km_s_Mpc <= 0:
                raise ValueError(f"Invalid H0={H0_km_s_Mpc}")
            return H0_km_s_Mpc * H0_CONVERSION, Omega_m, Omega_Lambda, Omega_r, Omega_k
    except Exception as e:
        logger.error(f"FITS file validation failed: {e}")
        return None

def get_h0_values():
    """Prompt for custom H0 values at runtime."""
    try:
        h0_input = input("Enter H0 values (km/s/Mpc) as comma-separated list or range (start:end:step, default 68): ").strip()
        if not h0_input:
            return [68]  # Default single value
        if ':' in h0_input:
            # Parse range (e.g., 60:80:5 for 60 to 80 with step 5)
            start, end, step = map(float, h0_input.split(':'))
            return np.arange(start, end + step, step).tolist()
        else:
            # Parse comma-separated list
            return [float(x) for x in h0_input.split(',')]
    except ValueError as e:
        logger.error(f"Invalid H0 input: {e}, using default H0=68")
        return [68]

def main():
    """Main function to run the quantum-classical conversion script with custom H0 values."""
    results = []
    try:
        N_input = input("Enter number of particles N (integer >= 2, default 2): ").strip()
        N = int(N_input) if N_input else 2
        if N < 2:
            raise ValueError("N must be >= 2")
        if N > MAX_N:
            logger.warning(f"N={N} exceeds MAX_N={MAX_N}, may cause numerical issues")
    except ValueError as e:
        logger.error(f"Invalid N input: {e}")
        return

    default_n_qubits = min(3, int(np.log2(N))) if N > 2 else 3
    try:
        n_qubits_input = input(f"Enter number of qubits (1 to {MAX_QUBITS}, default {default_n_qubits}): ").strip()
        n_qubits = int(n_qubits_input) if n_qubits_input else default_n_qubits
        if n_qubits < 1 or n_qubits > MAX_QUBITS:
            raise ValueError(f"n_qubits must be between 1 and {MAX_QUBITS}")
    except ValueError as e:
        logger.error(f"Invalid n_qubits input: {e}")
        return
    logger.info(f"Selected n_qubits={n_qubits} for N={N}")

    # Get custom H0 values
    h0_values = get_h0_values()
    logger.info(f"H0 values to test: {h0_values}")

    include_cosmological = input("Include cosmological calculations? (yes/no, default no): ").lower()
    t, z = None, None
    Omega_m, Omega_Lambda, Omega_r, Omega_k = Omega_m_default, Omega_Lambda_default, Omega_r_default, Omega_k_default

    if include_cosmological in ['yes', 'y']:
        use_cmb = input("Use CMB data from FITS file? (yes/no, default no): ").lower()
        if use_cmb in ['yes', 'y']:
            fits_file = input("Enter path to CMB FITS file (default /home/bbear/QT/hybrid_bao_cmb_sh0es_params.fits): ").strip() or '/home/bbear/QT/hybrid_bao_cmb_sh0es_params.fits'
            fits_params = validate_fits_file(fits_file)
            if fits_params:
                _, Omega_m, Omega_Lambda, Omega_r, Omega_k = fits_params
                logger.info(f"Loaded FITS parameters: Omega_m={Omega_m:.4f}, Omega_Lambda={Omega_Lambda:.4f}, Omega_r={Omega_r:.4e}, Omega_k={Omega_k:.4e}")
            else:
                logger.info("Using default parameters: Omega_m=0.3, Omega_Lambda=0.7, Omega_r=9.05e-5")

        try:
            t_input = input("Enter lookback time t in years (default 1e9): ").strip()
            t = float(t_input) if t_input else 1e9
            if t < 0:
                raise ValueError("t must be >= 0")
            auto_z = input("Calculate redshift z automatically? (yes/no, default yes): ").lower()
            if auto_z in ['yes', 'y', '']:
                z_values = []
                for h0_km_s_Mpc in h0_values:
                    h0 = h0_km_s_Mpc * H0_CONVERSION
                    z = compute_redshift(t, lookback=True, H0_value=h0, Omega_r=Omega_r, Omega_m=Omega_m, Omega_k=Omega_k, Omega_Lambda=Omega_Lambda)
                    z_values.append(z)
                    logger.info(f"For H0={h0_km_s_Mpc:.2f} km/s/Mpc, calculated redshift z={z:.6f}")
            else:
                z_input = input("Enter redshift z (default 0.1): ").strip()
                z = float(z_input) if z_input else 0.1
                if z < 0:
                    raise ValueError("z must be >= 0")
                z_values = [z] * len(h0_values)
        except ValueError as e:
            logger.error(f"Invalid cosmological input: {e}")
            return
    else:
        z_values = [None] * len(h0_values)

    S_quantum, M_quantum, circuit, counts, meas_expectation = compute_quantum_qiskit(N, n_qubits=n_qubits)
    S_classical, M_classical = compute_classical(N)
    rho, purity, expectation_Z = compute_density_matrix(circuit, z_values[0] if include_cosmological in ['yes', 'y'] else None)
    
    if abs(expectation_Z - meas_expectation) > 0.5:
        logger.warning(f"Discrepancy between expectation_Z={expectation_Z:.6f} and meas_expectation={meas_expectation:.6f}")

    # Store results for each H0
    for h0_km_s_Mpc, z in zip(h0_values, z_values):
        result = {
            'N': N,
            'n_qubits': n_qubits,
            'H0_km_s_Mpc': h0_km_s_Mpc,
            'S_quantum': S_quantum,
            'M_quantum': M_quantum,
            'S_classical': S_classical,
            'M_classical': M_classical,
            'circuit_counts': counts,
            'circuit_diagram': str(circuit.draw()),
            'purity': purity,
            'expectation_Z': expectation_Z,
            'meas_expectation': meas_expectation
        }
        if t is not None and z is not None:
            c_eff, N_DM = compute_cosmological(t, z)
            cosmo_result = {
                'H0_km_s_Mpc': h0_km_s_Mpc,
                'z': z,
                'c_eff': c_eff,
                'N_DM': N_DM
            }
            result.update(cosmo_result)
        results.append(result)

    # Save combined results
    output_file = f'qconvert_results_N{N}_nqubits{n_qubits}.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Results for N={N}, n_qubits={n_qubits}\n")
        for result in results:
            f.write(f"\n--- H0={result['H0_km_s_Mpc']:.2f} km/s/Mpc ---\n")
            f.write(f"Quantum S_quantum: {result['S_quantum']:.4f}\n")
            f.write(f"Quantum M_quantum: {result['M_quantum']:.4f}\n")
            f.write(f"Classical S_classical: {result['S_classical']:.4f}\n")
            f.write(f"Classical M_classical: {result['M_classical']:.4f}\n")
            f.write(f"Quantum circuit counts: {result['circuit_counts']}\n")
            f.write(f"Circuit diagram:\n{result['circuit_diagram']}\n")
            f.write(f"Density matrix purity: {result['purity']:.6f}\n")
            f.write(f"Density matrix Z expectation value: {result['expectation_Z']:.6f}\n")
            if 'z' in result:
                f.write(f"Density matrix scaled by (1+z)^3: {(1+result['z'])**3:.6f}\n")
                f.write(f"Redshift z: {result['z']:.6f}\n")
                f.write(f"Effective speed of light c_eff: {result['c_eff']:.4e} m/s\n")
                f.write(f"Dark matter number density N_DM: {result['N_DM']:.4e} m^-3\n")
    
    logger.info(f"Results saved to {output_file}")

    # Generate redshift histogram from FITS file
    if include_cosmological in ['yes', 'y']:
        try:
            with fits.open('/home/bbear/QT/hybrid_bao_cmb_sh0es_params.fits') as hdulist:
                z_data = hdulist['BAO_CATALOG'].data['Z'][:100]  # First 100 for histogram
                np.savetxt('/home/bbear/QT/z_data.txt', z_data, fmt='%.6f')
                logger.info(f"Saved first 100 redshift values to /home/bbear/QT/z_data.txt")
        except Exception as e:
            logger.error(f"Failed to generate redshift histogram data: {e}")

if __name__ == "__main__":
    main()