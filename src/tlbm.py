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
from typing import Tuple, Dict, Optional
from .config import TLBMConfig


class TLBM_MLX:
    """MLX-based Thermal Lattice Boltzmann Method solver"""

    def __init__(self, nx: int, ny: int, config: TLBMConfig, debug: bool = True):
        # Store physical domain size
        self.nx_phys = nx
        self.ny_phys = ny
        # Total size includes ghost nodes
        self.nx = nx + 2  # Add ghost layer on each side
        self.ny = ny + 2  # Add ghost layer on each side
        
        # Define slices for physical domain
        self.phys_x = slice(1, self.nx - 1)
        self.phys_y = slice(1, self.ny - 1)
        
        self.config = config
        self.debug = debug
        self.step_counter = 0

        # Use pre-computed relaxation times from config
        self.tau_f = config.tau_f
        self.tau_t = config.tau_g  # tau_t is for temperature
        self.tau_r = config.tau_r

        if self.debug:
            print(f"\nTLBM Initialization:")
            print(f"  Physical grid: {self.nx_phys}×{self.ny_phys}")
            print(f"  Total grid (with ghost): {self.nx}×{self.ny}")
            print(f"  Relaxation times: τ_f={self.tau_f:.3f}, τ_t={self.tau_t:.3f}, τ_r={self.tau_r:.3f}")
            print(f"  Gravity: {config.gravity:.3f}")
            print(f"  Ra={config.Ra:.1e}, Pr={config.Pr:.2f}")

        # D2Q9 lattice weights and velocities
        self.w = mx.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.e = mx.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                          [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=mx.int32)

        # Initialize distribution functions (with ghost nodes)
        self.f = mx.zeros((self.nx, self.ny, 9), dtype=mx.float32)  # Fluid
        self.g = mx.zeros((self.nx, self.ny, 9), dtype=mx.float32)  # Temperature
        self.r = mx.zeros((self.nx, self.ny, 9), dtype=mx.float32)  # Resource

        # Macroscopic fields (with ghost nodes)
        self.rho = mx.ones((self.nx, self.ny), dtype=mx.float32)
        self.vel = mx.zeros((self.nx, self.ny, 2), dtype=mx.float32)
        self.T = mx.zeros((self.nx, self.ny), dtype=mx.float32)
        self.P = mx.zeros((self.nx, self.ny), dtype=mx.float32)  # Resource concentration

        # Auxiliary fields (physical domain only)
        self.cell_presence = mx.zeros((self.nx_phys, self.ny_phys), dtype=mx.float32)

        # Pre-compute streaming indices for efficiency
        self._compute_streaming_indices()

        # Initialize fields
        self._initialize_fields()

        # Compile the main functions
        self._compile_functions()

    def _compute_streaming_indices(self):
        """Pre-compute indices for streaming operation with ghost nodes"""
        # Create index arrays for entire domain including ghost
        idx_x = mx.arange(self.nx)[:, None]
        idx_y = mx.arange(self.ny)[None, :]

        # For streaming, we need to know where each site gets its values FROM
        # With ghost nodes, ALL cells have valid neighbors
        self.stream_src_idx = []
        for k in range(9):
            # Source indices (where values come FROM)
            # This is the opposite direction of the velocity
            # Ghost nodes handle periodicity in x naturally
            src_x = (idx_x - self.e[k, 0]) % self.nx  
            src_y = idx_y - self.e[k, 1]
            
            # Clamp src_y to valid range [0, ny-1]
            # Ghost nodes at y=0 and y=ny-1 will handle boundary conditions
            src_y = mx.clip(src_y, 0, self.ny - 1)
            
            # Flatten indices for gather operation
            flat_idx = src_x * self.ny + src_y
            self.stream_src_idx.append(flat_idx.flatten())

        # Verify streaming indices
        if self.debug:
            print("\nVerifying streaming indices (with ghost nodes)...")
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
        # Initialize physical domain only
        # Temperature field: linear gradient from hot (bottom) to cold (top)
        y_coords = mx.arange(self.ny_phys)[None, :] / (self.ny_phys - 1)
        T_phys = self.config.T_hot * (1 - y_coords) + self.config.T_cold * y_coords
        T_phys = mx.broadcast_to(T_phys, (self.nx_phys, self.ny_phys))
        
        # Add small random perturbation to trigger instability
        perturbation = 0.01 * (mx.random.uniform(shape=(self.nx_phys, self.ny_phys)) - 0.5)
        T_phys = T_phys + perturbation
        
        # Set physical domain temperature
        self.T[self.phys_x, self.phys_y] = T_phys

        # Resource field: source at bottom of physical domain
        self.P[self.phys_x, 1] = 1.0  # Bottom of physical domain (y=1)

        # Initialize density in physical domain
        self.rho[self.phys_x, self.phys_y] = 1.0

        # Initialize distributions at equilibrium for physical domain
        for k in range(9):
            self.f[self.phys_x, self.phys_y, k] = self.w[k] * self.rho[self.phys_x, self.phys_y]
            self.g[self.phys_x, self.phys_y, k] = self.w[k] * self.T[self.phys_x, self.phys_y]
            self.r[self.phys_x, self.phys_y, k] = self.w[k] * self.P[self.phys_x, self.phys_y]
        
        # Initialize ghost nodes (will be set by boundary conditions)
        self._apply_ghost_boundaries()

        if self.debug:
            print("\nInitial state check (physical domain):")
            T_phys_check = self.T[self.phys_x, self.phys_y]
            P_phys_check = self.P[self.phys_x, self.phys_y]
            rho_phys_check = self.rho[self.phys_x, self.phys_y]
            f_phys_check = self.f[self.phys_x, self.phys_y, :]
            print(f"  T range: [{float(mx.min(T_phys_check)):.3f}, {float(mx.max(T_phys_check)):.3f}]")
            print(f"  P range: [{float(mx.min(P_phys_check)):.3f}, {float(mx.max(P_phys_check)):.3f}]")
            print(f"  ρ range: [{float(mx.min(rho_phys_check)):.3f}, {float(mx.max(rho_phys_check)):.3f}]")
            print(f"  f sum: {float(mx.sum(f_phys_check)):.3f} (should be {self.nx_phys * self.ny_phys})")
            print(f"  Total f sum (with ghost): {float(mx.sum(self.f)):.3f}")

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
            # Get shape from input arrays
            nx, ny = rho.shape
            vel_x = mx.broadcast_to(vel_x[:, :, :, None], (nx, ny, 1, 9))
            vel_y = mx.broadcast_to(vel_y[:, :, :, None], (nx, ny, 1, 9))

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
        def fused_collide_stream_with_force(f, g, r, rho, vel, T, P, w, e, tau_f, tau_t, tau_r, 
                                           force_y, stream_src_idx):
            """Fused collision and streaming with Guo forcing"""
            # Compute equilibrium distributions
            f_eq, g_eq, r_eq = compute_equilibrium(rho, vel, T, P, w, e)

            # Apply BGK collision
            f_post = f + (f_eq - f) / tau_f
            g_post = g + (g_eq - g) / tau_t
            r_post = r + (r_eq - r) / tau_r
            
            # Apply Guo forcing to momentum equation (vectorized)
            # Fi = (1 - 1/(2*tau)) * wi * 3 * (ei · F)
            force_factor = 1.0 - 0.5 / tau_f
            
            # Vectorized computation of forcing term
            # e[:, 1] is the y-component of lattice velocities
            # Shape: force_y is (nx, ny), need to broadcast
            force_y_expanded = mx.expand_dims(force_y, axis=2)  # (nx, ny, 1)
            e_y = e[:, 1].reshape(1, 1, 9)  # (1, 1, 9)
            w_reshaped = w.reshape(1, 1, 9)  # (1, 1, 9)
            
            # Compute all forcing terms at once
            Fi = force_factor * 3.0 * w_reshaped * e_y * force_y_expanded
            f_post = f_post + Fi

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

        self.fused_collide_stream_with_force = fused_collide_stream_with_force

    def compute_macroscopic(self):
        """Compute macroscopic quantities from distributions"""
        # Density and velocity from fluid distributions
        self.rho = mx.sum(self.f, axis=2)
        
        # Only check physical domain values
        rho_phys = self.rho[self.phys_x, self.phys_y]

        # Check for density issues (physical domain only)
        if self.debug and self.step_counter % 100 == 0:
            rho_min = float(mx.min(rho_phys))
            rho_max = float(mx.max(rho_phys))
            if rho_min < 0.5 or rho_max > 2.0:
                print(f"\n⚠️  Step {self.step_counter}: Density out of bounds!")
                print(f"   ρ range: [{rho_min:.3f}, {rho_max:.3f}]")

        # Momentum
        mom_x = mx.sum(self.f * self.e[:, 0].reshape(1, 1, 9), axis=2)
        mom_y = mx.sum(self.f * self.e[:, 1].reshape(1, 1, 9), axis=2)

        # Velocity
        self.vel = mx.stack([mom_x / self.rho, mom_y / self.rho], axis=2)

        # Check for velocity issues (physical domain only)
        if self.debug and self.step_counter % 100 == 0:
            vel_phys = self.vel[self.phys_x, self.phys_y, :]
            vel_mag = mx.sqrt(vel_phys[:, :, 0]**2 + vel_phys[:, :, 1]**2)
            vel_max = float(mx.max(vel_mag))
            if vel_max > 0.3:  # Mach number should be < 0.3 for LBM
                print(f"\n⚠️  Step {self.step_counter}: Velocity too high!")
                print(f"   Max velocity: {vel_max:.3f} (Mach={vel_max/0.577:.3f})")

        # Apply cell resistance (only in physical domain)
        if mx.any(self.cell_presence > 0):
            resistance_factor = mx.ones((self.nx, self.ny))
            resistance_factor[self.phys_x, self.phys_y] = (
                1.0 - self.config.cell_resistance * self.cell_presence
            )
            self.vel *= mx.expand_dims(resistance_factor, axis=2)

        # Temperature and resource
        self.T = mx.sum(self.g, axis=2)
        self.P = mx.sum(self.r, axis=2)

        # Check temperature bounds (physical domain only)
        if self.debug and self.step_counter % 100 == 0:
            T_phys = self.T[self.phys_x, self.phys_y]
            T_min = float(mx.min(T_phys))
            T_max = float(mx.max(T_phys))
            if T_min < -0.1 or T_max > 1.1:
                print(f"\n⚠️  Step {self.step_counter}: Temperature out of bounds!")
                print(f"   T range: [{T_min:.3f}, {T_max:.3f}]")

    # Note: streaming_step is no longer used - we use fused_collide_stream instead
    # Keeping this for reference but it's not called
    def streaming_step_old(self):
        """[DEPRECATED] Perform streaming using scatter operations - now fused with collision"""
        pass

    def _apply_ghost_boundaries(self):
        """Set ghost node values to enforce boundary conditions"""
        # Bottom ghost nodes (y=0): prepare for bounce-back
        # When streamed, these will enforce no-slip at y=1
        self.f[:, 0, 2] = self.f[:, 1, 4]   # North at ghost = South at wall
        self.f[:, 0, 5] = self.f[:, 1, 7]   # NE at ghost = SW at wall  
        self.f[:, 0, 6] = self.f[:, 1, 8]   # NW at ghost = SE at wall
        
        # Top ghost nodes (y=ny-1): prepare for bounce-back
        # When streamed, these will enforce no-slip at y=ny-2
        self.f[:, -1, 4] = self.f[:, -2, 2]  # South at ghost = North at wall
        self.f[:, -1, 7] = self.f[:, -2, 5]  # SW at ghost = NE at wall
        self.f[:, -1, 8] = self.f[:, -2, 6]  # SE at ghost = NW at wall
        
        # Temperature boundary conditions
        # Bottom physical boundary (y=1) should have T_hot
        self.T[:, 1] = self.config.T_hot
        # Top physical boundary (y=ny-2) should have T_cold  
        self.T[:, -2] = self.config.T_cold
        
        # Resource boundary conditions
        self.P[:, 1] = 1.0   # Source at bottom
        self.P[:, -2] = 0.0  # Sink at top
        
        # Set ghost node distributions for temperature and resource (vectorized)
        # Use broadcasting to set all directions at once
        w_broadcast = self.w.reshape(1, 1, 9)
        
        # Bottom boundaries (ghost and physical)
        self.g[:, 0:2, :] = w_broadcast * self.config.T_hot
        self.r[:, 0:2, :] = w_broadcast * 1.0
        
        # Top boundaries (ghost and physical)
        self.g[:, -2:, :] = w_broadcast * self.config.T_cold
        self.r[:, -2:, :] = w_broadcast * 0.0
        
        # Handle x-periodic ghost nodes
        # Left ghost (x=0) = right physical (x=nx-2)
        self.f[0, :, :] = self.f[-2, :, :]
        self.g[0, :, :] = self.g[-2, :, :]
        self.r[0, :, :] = self.r[-2, :, :]
        
        # Right ghost (x=nx-1) = left physical (x=1)
        self.f[-1, :, :] = self.f[1, :, :]
        self.g[-1, :, :] = self.g[1, :, :]
        self.r[-1, :, :] = self.r[1, :, :]

    def _compute_force_field(self):
        """Compute buoyancy force field (for Guo forcing)"""
        T_ref = 0.5 * (self.config.T_hot + self.config.T_cold)
        # Compute force only in physical domain
        self.force_y = mx.zeros((self.nx, self.ny))
        self.force_y[self.phys_x, self.phys_y] = (
            self.config.gravity * self.config.beta * 
            (self.T[self.phys_x, self.phys_y] - T_ref)
        )
        
        # Debug force magnitude
        if self.debug and self.step_counter % 500 == 0:
            force_phys = self.force_y[self.phys_x, self.phys_y]
            force_max = float(mx.max(mx.abs(force_phys)))
            print(f"\nStep {self.step_counter}: Max buoyancy force = {force_max:.6f}")

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

        # 1. Apply ghost boundary conditions BEFORE streaming
        self._apply_ghost_boundaries()
        
        # 2. Compute macroscopic quantities
        self.compute_macroscopic()

        # 3. Compute force field (for Guo forcing)
        self._compute_force_field()

        # 4. Fused collision and streaming with Guo forcing
        self.f, self.g, self.r = self.fused_collide_stream_with_force(
            self.f, self.g, self.r,
            self.rho, self.vel, self.T, self.P,
            self.w, self.e,
            self.tau_f, self.tau_t, self.tau_r,
            self.force_y,
            self.stream_src_idx
        )

        # Check distribution sums for conservation (physical domain only)
        if self.debug and self.step_counter % 500 == 0:
            f_phys = self.f[self.phys_x, self.phys_y, :]
            f_sum = float(mx.sum(f_phys))
            print(f"Step {self.step_counter}: After fused collide-stream, f sum = {f_sum:.6f} (should = {self.nx_phys*self.ny_phys})")

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
        # Temperature gradient at bottom boundary (physical domain)
        # Bottom physical boundary is at y=1, next point is y=2
        dT_dy = (self.T[self.phys_x, 2] - self.T[self.phys_x, 1])
        avg_gradient = mx.mean(dT_dy)

        # Nusselt = heat flux / conductive flux
        Nu = abs(avg_gradient) * self.ny_phys / (self.config.T_hot - self.config.T_cold)

        if self.debug and self.step_counter % 500 == 0:
            print(f"Step {self.step_counter}: Nu = {float(Nu):.3f}, avg gradient = {float(avg_gradient):.6f}")

        return float(Nu)

    def get_numpy_fields(self) -> Dict[str, np.ndarray]:
        """Convert fields to numpy for visualization (physical domain only)"""
        mx.eval(self.T, self.vel, self.P, self.rho)  # Ensure arrays are evaluated

        # Extract physical domain only
        return {
            'T': np.array(self.T[self.phys_x, self.phys_y]),
            'vx': np.array(self.vel[self.phys_x, self.phys_y, 0]),
            'vy': np.array(self.vel[self.phys_x, self.phys_y, 1]),
            'P': np.array(self.P[self.phys_x, self.phys_y]),
            'rho': np.array(self.rho[self.phys_x, self.phys_y])
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
    print(f"Grid size: 128×64")
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
