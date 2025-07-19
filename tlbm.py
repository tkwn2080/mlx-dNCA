"""
MLX-based Thermal Lattice Boltzmann Method (TLBM) implementation
Optimized for Apple Silicon using unified memory and JIT compilation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class TLBMConfig:
    """Configuration for TLBM simulation"""
    # Physical parameters
    Ra: float = 1e4          # Rayleigh number
    Pr: float = 0.71         # Prandtl number
    T_hot: float = 1.0       # Hot boundary temperature
    T_cold: float = 0.0      # Cold boundary temperature

    # Derived parameters (computed in __post_init__)
    niu: float = None        # Kinematic viscosity
    kappa: float = None      # Thermal diffusivity
    gravity: float = None    # Gravity magnitude
    beta: float = 0.5        # Thermal expansion coefficient - reduced for stability

    # Resource parameters
    resource_kappa: float = 0.05  # Resource diffusivity - increased for stability
    cell_resistance: float = 0.5   # Flow resistance from cells

    def __post_init__(self):
        """Compute derived parameters from Ra and Pr"""
        # For a domain of height H=1 in lattice units
        # Ra = g*beta*DT*H^3/(niu*kappa)
        # Pr = niu/kappa

        # Choose kappa and compute niu
        self.kappa = 0.1
        self.niu = self.Pr * self.kappa

        # Use stable gravity value instead of computing from Ra
        # Computing from Ra gives unrealistically high values that cause instability
        # For Ra=2e4, the computed gravity would be ~142, which is way too high
        # Use a small value similar to successful implementations
        # Common LBM implementations use gravity values around 0.001-0.01
        self.gravity = 0.005  # Moderate value for visible convection

        # Compute effective Ra for display
        DT = self.T_hot - self.T_cold
        H = 1.0  # Height in lattice units
        Ra_effective = self.gravity * self.beta * DT * H**3 / (self.niu * self.kappa)

        print(f"Using gravity: {self.gravity:.6f}")
        print(f"Effective Ra: {Ra_effective:.1e} (requested: {self.Ra:.1e})")


class TLBM_MLX:
    """MLX-based Thermal Lattice Boltzmann Method solver"""

    def __init__(self, nx: int, ny: int, config: TLBMConfig, debug: bool = True):
        self.nx = nx
        self.ny = ny
        self.config = config
        self.debug = debug
        self.step_counter = 0

        # Compute relaxation times
        self.tau_f = 3.0 * config.niu + 0.5
        self.tau_t = 3.0 * config.kappa + 0.5
        self.tau_r = 3.0 * config.resource_kappa + 0.5

        if self.debug:
            print(f"\nTLBM Initialization:")
            print(f"  Grid: {nx}×{ny}")
            print(f"  Relaxation times: τ_f={self.tau_f:.3f}, τ_t={self.tau_t:.3f}, τ_r={self.tau_r:.3f}")
            print(f"  Gravity: {config.gravity:.3f}")
            print(f"  Ra={config.Ra:.1e}, Pr={config.Pr:.2f}")

        # D2Q9 lattice weights and velocities
        self.w = mx.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.e = mx.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                          [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=mx.int32)

        # Initialize distribution functions
        self.f = mx.zeros((nx, ny, 9), dtype=mx.float32)  # Fluid
        self.g = mx.zeros((nx, ny, 9), dtype=mx.float32)  # Temperature
        self.r = mx.zeros((nx, ny, 9), dtype=mx.float32)  # Resource

        # Macroscopic fields
        self.rho = mx.ones((nx, ny), dtype=mx.float32)
        self.vel = mx.zeros((nx, ny, 2), dtype=mx.float32)
        self.T = mx.zeros((nx, ny), dtype=mx.float32)
        self.P = mx.zeros((nx, ny), dtype=mx.float32)  # Resource concentration

        # Auxiliary fields
        self.cell_presence = mx.zeros((nx, ny), dtype=mx.float32)
        self.mask = mx.zeros((nx, ny), dtype=mx.float32)  # Solid boundaries

        # Pre-compute streaming indices for efficiency
        self._compute_streaming_indices()

        # Initialize fields
        self._initialize_fields()

        # Compile the main functions
        self._compile_functions()

    def _compute_streaming_indices(self):
        """Pre-compute indices for streaming operation"""
        # Create index arrays
        idx_x = mx.arange(self.nx)[:, None]
        idx_y = mx.arange(self.ny)[None, :]

        # For streaming, we need to know where each site gets its values FROM
        # This is the opposite of the velocity direction
        self.stream_src_idx = []
        for k in range(9):
            # Source indices (where values come FROM)
            # This is the opposite direction of the velocity
            src_x = (idx_x - self.e[k, 0]) % self.nx  # Periodic in x
            src_y = idx_y - self.e[k, 1]

            # Handle y boundaries (non-periodic)
            src_y = mx.clip(src_y, 0, self.ny - 1)

            # Flatten indices for gather operation
            flat_idx = src_x * self.ny + src_y
            self.stream_src_idx.append(flat_idx.flatten())

        # Verify streaming indices
        if self.debug:
            print("\nVerifying streaming indices...")
            for k in range(9):
                idx_min = int(mx.min(self.stream_src_idx[k]))
                idx_max = int(mx.max(self.stream_src_idx[k]))
                expected_max = self.nx * self.ny - 1
                if idx_min < 0 or idx_max > expected_max:
                    print(f"  ❌ Direction {k}: indices out of bounds! [{idx_min}, {idx_max}] (expected [0, {expected_max}])")
                else:
                    print(f"  ✓ Direction {k}: indices OK [{idx_min}, {idx_max}]")

    def _initialize_fields(self):
        """Initialize temperature gradient and equilibrium distributions"""
        # Temperature field: linear gradient from hot (bottom) to cold (top)
        y_coords = mx.arange(self.ny)[None, :] / (self.ny - 1)
        self.T = self.config.T_hot * (1 - y_coords) + self.config.T_cold * y_coords
        self.T = mx.broadcast_to(self.T, (self.nx, self.ny))
        
        # Add small random perturbation to trigger instability
        perturbation = 0.01 * (mx.random.uniform(shape=(self.nx, self.ny)) - 0.5)
        self.T = self.T + perturbation

        # Resource field: source at bottom
        self.P = mx.zeros((self.nx, self.ny))
        self.P[:, 0] = 1.0  # Source at bottom

        # Initialize distributions at equilibrium
        for k in range(9):
            self.f[:, :, k] = self.w[k] * self.rho
            self.g[:, :, k] = self.w[k] * self.T
            self.r[:, :, k] = self.w[k] * self.P

        if self.debug:
            print("\nInitial state check:")
            print(f"  T range: [{float(mx.min(self.T)):.3f}, {float(mx.max(self.T)):.3f}]")
            print(f"  P range: [{float(mx.min(self.P)):.3f}, {float(mx.max(self.P)):.3f}]")
            print(f"  ρ range: [{float(mx.min(self.rho)):.3f}, {float(mx.max(self.rho)):.3f}]")
            print(f"  f sum: {float(mx.sum(self.f)):.3f} (should be {self.nx * self.ny})")

    def _compile_functions(self):
        """Compile main functions with mx.compile for JIT optimization"""
        # Note: mx.compile works best with pure functions
        # We'll compile the core computational kernels

        @mx.compile
        def compute_equilibrium(rho, vel, T, P, w, e):
            """Compute equilibrium distributions"""
            # Velocity dot products
            vel_x = vel[:, :, 0:1]
            vel_y = vel[:, :, 1:2]

            # Broadcast for all directions
            vel_x = mx.broadcast_to(vel_x[:, :, :, None], (self.nx, self.ny, 1, 9))
            vel_y = mx.broadcast_to(vel_y[:, :, :, None], (self.nx, self.ny, 1, 9))

            # e velocities
            ex = e[:, 0].reshape(1, 1, 1, 9)
            ey = e[:, 1].reshape(1, 1, 1, 9)

            # eu = e · u
            eu = ex * vel_x + ey * vel_y
            eu = eu.squeeze(2)  # Remove extra dimension

            # u · u
            u2 = vel[:, :, 0]**2 + vel[:, :, 1]**2
            u2 = mx.expand_dims(u2, axis=2)

            # Equilibrium distributions
            w_broadcast = w.reshape(1, 1, 9)

            # Fluid equilibrium
            rho_exp = mx.expand_dims(rho, axis=2)
            f_eq = w_broadcast * rho_exp * (1 + 3*eu + 4.5*eu**2 - 1.5*u2)

            # Temperature equilibrium
            T_exp = mx.expand_dims(T, axis=2)
            g_eq = w_broadcast * T_exp * (1 + 3*eu)

            # Resource equilibrium
            P_exp = mx.expand_dims(P, axis=2)
            r_eq = w_broadcast * P_exp * (1 + 3*eu)

            return f_eq, g_eq, r_eq

        self.compute_equilibrium = compute_equilibrium

        @mx.compile
        def apply_collision(f, g, r, f_eq, g_eq, r_eq, tau_f, tau_t, tau_r):
            """Apply BGK collision operator"""
            f_new = f + (f_eq - f) / tau_f
            g_new = g + (g_eq - g) / tau_t
            r_new = r + (r_eq - r) / tau_r
            return f_new, g_new, r_new

        self.apply_collision = apply_collision

        @mx.compile
        def fused_collide_stream(f, g, r, rho, vel, T, P, w, e, tau_f, tau_t, tau_r, stream_src_idx):
            """Fused collision and streaming operation for better memory efficiency"""
            # Compute equilibrium distributions
            f_eq, g_eq, r_eq = compute_equilibrium(rho, vel, T, P, w, e)

            # Apply collision
            f_post = f + (f_eq - f) / tau_f
            g_post = g + (g_eq - g) / tau_t
            r_post = r + (r_eq - r) / tau_r

            # Create new arrays for streamed values
            f_new = mx.zeros_like(f)
            g_new = mx.zeros_like(g)
            r_new = mx.zeros_like(r)

            # Stream each direction (fused with collision)
            for k in range(9):
                # Get post-collision values for this direction
                f_k = f_post[:, :, k].flatten()
                g_k = g_post[:, :, k].flatten()
                r_k = r_post[:, :, k].flatten()

                # Gather values from source locations
                # This gets values FROM the source indices
                f_streamed = f_k[stream_src_idx[k]]
                g_streamed = g_k[stream_src_idx[k]]
                r_streamed = r_k[stream_src_idx[k]]

                # Reshape and assign to new arrays
                f_new[:, :, k] = f_streamed.reshape(f.shape[0], f.shape[1])
                g_new[:, :, k] = g_streamed.reshape(g.shape[0], g.shape[1])
                r_new[:, :, k] = r_streamed.reshape(r.shape[0], r.shape[1])

            return f_new, g_new, r_new

        self.fused_collide_stream = fused_collide_stream

    def compute_macroscopic(self):
        """Compute macroscopic quantities from distributions"""
        # Density and velocity from fluid distributions
        self.rho = mx.sum(self.f, axis=2)

        # Check for density issues
        if self.debug and self.step_counter % 100 == 0:
            rho_min = float(mx.min(self.rho))
            rho_max = float(mx.max(self.rho))
            if rho_min < 0.5 or rho_max > 2.0:
                print(f"\n⚠️  Step {self.step_counter}: Density out of bounds!")
                print(f"   ρ range: [{rho_min:.3f}, {rho_max:.3f}]")

        # Momentum
        mom_x = mx.sum(self.f * self.e[:, 0].reshape(1, 1, 9), axis=2)
        mom_y = mx.sum(self.f * self.e[:, 1].reshape(1, 1, 9), axis=2)

        # Velocity
        self.vel = mx.stack([mom_x / self.rho, mom_y / self.rho], axis=2)

        # Check for velocity issues
        if self.debug and self.step_counter % 100 == 0:
            vel_mag = mx.sqrt(self.vel[:, :, 0]**2 + self.vel[:, :, 1]**2)
            vel_max = float(mx.max(vel_mag))
            if vel_max > 0.3:  # Mach number should be < 0.3 for LBM
                print(f"\n⚠️  Step {self.step_counter}: Velocity too high!")
                print(f"   Max velocity: {vel_max:.3f} (Mach={vel_max/0.577:.3f})")

        # Apply cell resistance
        resistance_factor = 1.0 - self.config.cell_resistance * self.cell_presence
        self.vel *= mx.expand_dims(resistance_factor, axis=2)

        # Temperature and resource
        self.T = mx.sum(self.g, axis=2)
        self.P = mx.sum(self.r, axis=2)

        # Check temperature bounds
        if self.debug and self.step_counter % 100 == 0:
            T_min = float(mx.min(self.T))
            T_max = float(mx.max(self.T))
            if T_min < -0.1 or T_max > 1.1:
                print(f"\n⚠️  Step {self.step_counter}: Temperature out of bounds!")
                print(f"   T range: [{T_min:.3f}, {T_max:.3f}]")

    # Note: streaming_step is no longer used - we use fused_collide_stream instead
    # Keeping this for reference but it's not called
    def streaming_step_old(self):
        """[DEPRECATED] Perform streaming using scatter operations - now fused with collision"""
        pass

    def apply_boundaries(self):
        """Apply boundary conditions"""
        # Bottom boundary (y=0): hot temperature, resource source, no-slip
        self.T[:, 0] = self.config.T_hot
        self.P[:, 0] = 1.0
        self.vel[:, 0, :] = 0.0  # No-slip boundary

        # Top boundary (y=ny-1): cold temperature, resource sink, no-slip
        self.T[:, -1] = self.config.T_cold
        self.P[:, -1] = 0.0
        self.vel[:, -1, :] = 0.0  # No-slip boundary

        # Apply bounce-back for no-slip walls at top and bottom
        # Bottom wall bounce-back
        self.f[:, 0, 2] = self.f[:, 0, 4]  # North -> South
        self.f[:, 0, 5] = self.f[:, 0, 7]  # NE -> SW
        self.f[:, 0, 6] = self.f[:, 0, 8]  # NW -> SE
        
        # Top wall bounce-back
        self.f[:, -1, 4] = self.f[:, -1, 2]  # South -> North
        self.f[:, -1, 7] = self.f[:, -1, 5]  # SW -> NE
        self.f[:, -1, 8] = self.f[:, -1, 6]  # SE -> NW

        # Update temperature distributions at boundaries
        for k in range(9):
            # Bottom
            self.g[:, 0, k] = self.w[k] * self.config.T_hot
            self.r[:, 0, k] = self.w[k] * 1.0

            # Top
            self.g[:, -1, k] = self.w[k] * self.config.T_cold
            self.r[:, -1, k] = self.w[k] * 0.0

    def apply_buoyancy_force(self):
        """Apply buoyancy force to velocity field"""
        T_ref = 0.5 * (self.config.T_hot + self.config.T_cold)
        # Remove grid-size scaling - the physics should be independent of grid resolution
        force_y = self.config.gravity * self.config.beta * (self.T - T_ref)

        # Debug force magnitude
        if self.debug and self.step_counter % 500 == 0:
            force_max = float(mx.max(mx.abs(force_y)))
            print(f"\nStep {self.step_counter}: Max buoyancy force = {force_max:.6f}")

        # Add force to y-velocity
        self.vel[:, :, 1] = self.vel[:, :, 1] + force_y

    def step(self):
        """Perform one complete TLBM step"""
        self.step_counter += 1

        # Check for NaN/Inf more frequently when debugging
        if self.debug and (self.step_counter <= 10 or self.step_counter % 10 == 0):
            if self._check_for_numerical_issues():
                print(f"Stopping simulation due to numerical issues at step {self.step_counter}")
                # Print additional diagnostic info
                self._diagnose_nan_source()
                return False  # Signal to stop

        # 1. Compute macroscopic quantities
        self.compute_macroscopic()

        # 2. Apply buoyancy force
        self.apply_buoyancy_force()

        # 3. Fused collision and streaming (as recommended by Stavros)
        self.f, self.g, self.r = self.fused_collide_stream(
            self.f, self.g, self.r,
            self.rho, self.vel, self.T, self.P,
            self.w, self.e,
            self.tau_f, self.tau_t, self.tau_r,
            self.stream_src_idx
        )

        # Check distribution sums for conservation
        if self.debug and self.step_counter % 500 == 0:
            f_sum = float(mx.sum(self.f))
            print(f"Step {self.step_counter}: After fused collide-stream, f sum = {f_sum:.6f} (should ≈ {self.nx*self.ny})")

        # 6. Apply boundaries
        self.apply_boundaries()

        return True  # Success

    def _check_for_numerical_issues(self):
        """Check for NaN or Inf in key fields"""
        fields_to_check = [
            ('f', self.f),
            ('g', self.g),
            ('r', self.r),
            ('rho', self.rho),
            ('vel', self.vel),
            ('T', self.T),
            ('P', self.P)
        ]

        for name, field in fields_to_check:
            if mx.any(mx.isnan(field)) or mx.any(mx.isinf(field)):
                print(f"\n❌ Step {self.step_counter}: NaN/Inf detected in {name}!")
                # Find location of issue
                nan_mask = mx.isnan(field)
                inf_mask = mx.isinf(field)
                if mx.any(nan_mask):
                    print(f"   NaN locations: {mx.sum(nan_mask)} values")
                if mx.any(inf_mask):
                    print(f"   Inf locations: {mx.sum(inf_mask)} values")
                return True
        return False

    def _diagnose_nan_source(self):
        """Diagnose where NaN values are coming from"""
        print("\n=== NaN Diagnostic ===")

        # Check each distribution
        for k in range(9):
            f_k = self.f[:, :, k]
            g_k = self.g[:, :, k]
            r_k = self.r[:, :, k]

            if mx.any(mx.isnan(f_k)):
                print(f"NaN in f distribution, direction {k}")
                # Find first NaN location
                nan_mask = mx.isnan(f_k)
                # MLX where requires 3 arguments, so we'll use a different approach
                # Convert to numpy to find indices
                nan_locations = np.argwhere(np.array(nan_mask))
                if len(nan_locations) > 0:
                    i, j = nan_locations[0]
                    print(f"  First NaN at ({i}, {j})")
                    print(f"  Neighboring values:")
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.nx and 0 <= nj < self.ny:
                                print(f"    f[{ni},{nj},{k}] = {float(self.f[ni, nj, k])}")

        # Check velocities
        vel_mag = mx.sqrt(self.vel[:, :, 0]**2 + self.vel[:, :, 1]**2)
        max_vel = float(mx.max(vel_mag[~mx.isnan(vel_mag)]))
        print(f"\nMax velocity (excluding NaN): {max_vel:.6f}")

        # Check relaxation times
        print(f"\nRelaxation times:")
        print(f"  τ_f = {self.tau_f:.3f}")
        print(f"  τ_t = {self.tau_t:.3f}")
        print(f"  τ_r = {self.tau_r:.3f}")

        print("===================\n")

    def calculate_nusselt(self) -> float:
        """Calculate Nusselt number"""
        # Temperature gradient at bottom boundary
        dT_dy = (self.T[:, 1] - self.T[:, 0])
        avg_gradient = mx.mean(dT_dy)

        # Nusselt = heat flux / conductive flux
        Nu = abs(avg_gradient) * self.ny / (self.config.T_hot - self.config.T_cold)

        if self.debug and self.step_counter % 500 == 0:
            print(f"Step {self.step_counter}: Nu = {float(Nu):.3f}, avg gradient = {float(avg_gradient):.6f}")

        return float(Nu)

    def get_numpy_fields(self) -> Dict[str, np.ndarray]:
        """Convert fields to numpy for visualization"""
        mx.eval(self.T, self.vel, self.P)  # Ensure arrays are evaluated

        return {
            'T': np.array(self.T),
            'vx': np.array(self.vel[:, :, 0]),
            'vy': np.array(self.vel[:, :, 1]),
            'P': np.array(self.P),
            'rho': np.array(self.rho)
        }


def visualize_tlbm(tlbm: TLBM_MLX, num_steps: int = 10000, update_interval: int = 100):
    """Run TLBM simulation with real-time visualization"""

    # Setup figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Temperature plot
    fields = tlbm.get_numpy_fields()
    im1 = ax1.imshow(fields['T'].T, cmap='hot', origin='lower',
                     vmin=tlbm.config.T_cold, vmax=tlbm.config.T_hot)
    ax1.set_title('Temperature Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)

    # Vorticity plot
    vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
    im2 = ax2.imshow(vort.T, cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
    ax2.set_title('Vorticity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    # Resource plot
    im3 = ax3.imshow(fields['P'].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
    ax3.set_title('Resource Concentration')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)

    # Nusselt number evolution
    steps_history = []
    nu_history = []
    line, = ax4.plot([], [], 'b-')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Nusselt Number')
    ax4.set_title('Heat Transfer Evolution')
    ax4.grid(True)

    # Add text for parameters
    param_text = (f"Ra = {tlbm.config.Ra:.1e}, Pr = {tlbm.config.Pr:.2f}\n"
                 f"Grid: {tlbm.nx}×{tlbm.ny}")
    fig.text(0.02, 0.02, param_text, fontsize=10)

    # Progress tracking
    start_time = time.time()

    def update(frame):
        # Run multiple steps between updates
        for _ in range(update_interval):
            tlbm.step()

        current_step = frame * update_interval

        # Get updated fields
        fields = tlbm.get_numpy_fields()

        # Update temperature
        im1.set_array(fields['T'].T)

        # Update vorticity
        vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
        im2.set_array(vort.T)

        # Update resource
        im3.set_array(fields['P'].T)

        # Calculate and plot Nusselt number
        nu = tlbm.calculate_nusselt()
        steps_history.append(current_step)
        nu_history.append(nu)

        line.set_data(steps_history, nu_history)
        ax4.relim()
        ax4.autoscale_view()

        # Performance info
        elapsed = time.time() - start_time
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0

        # Update title with progress
        fig.suptitle(f'TLBM Simulation - Step {current_step}/{num_steps} '
                    f'({steps_per_sec:.1f} steps/s) - Nu = {nu:.3f}')

        return [im1, im2, im3, line]

    # Create animation
    frames = num_steps // update_interval
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    return anim


def test_benard_convection():
    """Test Benard convection formation with MLX TLBM"""

    print("MLX TLBM - Benard Convection Test")
    print("==================================")

    # Create configuration
    config = TLBMConfig(
        Ra=2e4,      # Above critical Rayleigh number
        Pr=0.71,     # Air-like fluid
        T_hot=1.0,
        T_cold=0.0,
        resource_kappa=0.02
    )

    # Print parameters
    print(f"Rayleigh number: {config.Ra:.1e}")
    print(f"Prandtl number: {config.Pr:.2f}")
    print(f"Grid size: 256×128")
    print(f"Relaxation times: τ_f={3*config.niu+0.5:.3f}, τ_t={3*config.kappa+0.5:.3f}")
    print(f"Gravity: {config.gravity:.3f}")
    print()

    # Create TLBM instance - start with smaller grid for stability testing
    print("Initializing TLBM...")
    tlbm = TLBM_MLX(nx=128, ny=64, config=config, debug=True)

    # Check device
    print(f"Using device: {mx.default_device()}")
    print()

    # Run with visualization
    print("Starting simulation...")
    print("Expected behavior:")
    print("- Steps 0-1000: Thermal diffusion")
    print("- Steps 1000-3000: Instability onset")
    print("- Steps 3000-7000: Convection rolls form")
    print("- Steps 7000-10000: Steady convection")
    print()

    visualize_tlbm(tlbm, num_steps=10000, update_interval=50)


if __name__ == "__main__":
    # Set default device
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)

    # Run test
    test_benard_convection()
