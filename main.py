"""
Main script for coupled TLBM-Cell system simulation
Visualizes fluid dynamics with cellular automata overlay
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time

from src.tlbm import TLBM_MLX
from src.cell import DenseCellSystem
from src.config import DEFAULT_CONFIG, SimulationConfig


def create_coupled_visualization(tlbm: TLBM_MLX, cells: DenseCellSystem, config: SimulationConfig,
                               num_steps: int = 50000, update_interval: int = 50):
    """Run coupled TLBM-Cell simulation with visualization"""

    # Setup figure with 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # Get initial fields
    fields = tlbm.get_numpy_fields()

    # 1. Temperature + Velocity field
    im1 = ax1.imshow(fields['T'].T, cmap='hot', origin='lower',
                     vmin=tlbm.config.T_cold, vmax=tlbm.config.T_hot, alpha=0.8)
    ax1.set_title('Temperature & Velocity Field')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Temperature')

    # Velocity vectors (quiver)
    Y, X = np.mgrid[0:tlbm.ny_phys:4, 0:tlbm.nx_phys:4]
    quiver = ax1.quiver(X, Y,
                       fields['vx'][::4, ::4].T,
                       fields['vy'][::4, ::4].T,
                       scale=3, alpha=0.6)

    # Cell scatter for temperature plot (size based on M)
    cell_pos = cells.get_cell_positions_numpy()
    M_values, A_values, B_values, E_values = cells.get_cell_values_numpy()
    if len(cell_pos) > 0:
        scatter1 = ax1.scatter(cell_pos[:, 0], cell_pos[:, 1],
                             s=M_values * 50, c='blue',
                             alpha=0.6, edgecolors='darkblue')
    else:
        scatter1 = ax1.scatter([], [], s=[], c='blue')

    # 2. Vorticity field (pre-computed in MLX)
    im2 = ax2.imshow(fields.get('vorticity', np.zeros_like(fields['vx'])).T, 
                     cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
    ax2.set_title('Vorticity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    # 3. Resource concentration with RGB visualization
    # Create RGB image: Red = A, Blue = B, Green = 0
    # Note: imshow expects (height, width, 3) so we transpose the fields
    rgb_resources = np.zeros((fields['P_A'].T.shape[0], fields['P_A'].T.shape[1], 3))
    rgb_resources[:, :, 0] = np.clip(fields['P_A'].T, 0, 1)  # Red channel for A
    rgb_resources[:, :, 2] = np.clip(fields['P_B'].T, 0, 1)  # Blue channel for B
    
    im3 = ax3.imshow(rgb_resources, origin='lower', alpha=0.8)
    ax3.set_title('Resource Distribution (Red=A, Blue=B)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    # Cell scatter for resource plot (color based on A/B ratio)
    if len(cell_pos) > 0:
        # Create cell colors based on their internal A/B content
        cell_colors = np.zeros((len(cell_pos), 3))
        cell_colors[:, 0] = np.clip(A_values / (A_values + B_values + 1e-6), 0, 1)  # Red
        cell_colors[:, 2] = np.clip(B_values / (A_values + B_values + 1e-6), 0, 1)  # Blue
        
        scatter3 = ax3.scatter(cell_pos[:, 0], cell_pos[:, 1],
                             s=E_values * 100, c=cell_colors,
                             alpha=0.7, edgecolors='black', linewidths=1)
    else:
        scatter3 = ax3.scatter([], [], s=[], c='darkgreen')

    # 4. System statistics
    ax4.set_xlim(0, num_steps)
    ax4.set_xlabel('Steps')
    ax4.set_title('System Evolution')
    ax4.grid(True, alpha=0.3)

    # Multiple y-axes for different quantities
    ax4_twin1 = ax4.twinx()
    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines['right'].set_position(('outward', 60))

    # Data storage
    steps_history = []
    n_cells_history = []
    avg_M_history = []
    nu_history = []

    # Plot lines
    line_cells, = ax4.plot([], [], 'b-', label='Cell Count')
    line_M, = ax4_twin1.plot([], [], 'r-', label='Avg M')
    line_nu, = ax4_twin2.plot([], [], 'g-', label='Nusselt')

    ax4.set_ylabel('Cell Count', color='b')
    ax4_twin1.set_ylabel('Average M', color='r')
    ax4_twin2.set_ylabel('Nusselt Number', color='g')

    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin1.tick_params(axis='y', labelcolor='r')
    ax4_twin2.tick_params(axis='y', labelcolor='g')

    # Add parameter text
    param_text = (f"TLBM: Ra={tlbm.config.Ra:.1e}, Pr={tlbm.config.Pr:.2f}\n"
                 f"Cells: rate1={cells.config.reaction_rate_1:.2f}, rate2={cells.config.reaction_rate_2:.2f}, "
                 f"div_thresh={cells.config.division_threshold:.1f}")
    fig.text(0.02, 0.02, param_text, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Progress tracking
    start_time = time.time()

    def update(frame):
        # Run multiple steps
        for _ in range(update_interval):
            # TLBM step
            if not tlbm.step():
                return []

            # Cell step (every N TLBM steps for stability)
            if tlbm.step_counter % config.coupling.cell_update_interval == 0:
                # Update cells with resource fields
                P_A_field = tlbm.P_A[tlbm.phys_x, tlbm.phys_y]
                P_B_field = tlbm.P_B[tlbm.phys_x, tlbm.phys_y]
                cells.step(P_A_field, P_B_field)

                # Apply cell effects to TLBM
                # 1. Update cell presence for flow resistance
                tlbm.cell_presence = cells.cell_presence

                # 2. Extract resources A and B
                if mx.any(cells.resource_A_extraction > 0):
                    P_A_phys = tlbm.P_A[tlbm.phys_x, tlbm.phys_y]
                    P_A_new = mx.maximum(0, P_A_phys - cells.resource_A_extraction * config.coupling.resource_extraction_factor)
                    tlbm.P_A[tlbm.phys_x, tlbm.phys_y] = P_A_new

                    # Update resource A distributions
                    for k in range(9):
                        tlbm.r_A[tlbm.phys_x, tlbm.phys_y, k] = (
                            tlbm.w[k] * tlbm.P_A[tlbm.phys_x, tlbm.phys_y]
                        )

                if mx.any(cells.resource_B_extraction > 0):
                    P_B_phys = tlbm.P_B[tlbm.phys_x, tlbm.phys_y]
                    P_B_new = mx.maximum(0, P_B_phys - cells.resource_B_extraction * config.coupling.resource_extraction_factor)
                    tlbm.P_B[tlbm.phys_x, tlbm.phys_y] = P_B_new

                    # Update resource B distributions
                    for k in range(9):
                        tlbm.r_B[tlbm.phys_x, tlbm.phys_y, k] = (
                            tlbm.w[k] * tlbm.P_B[tlbm.phys_x, tlbm.phys_y]
                        )

        current_step = frame * update_interval

        # Get updated fields
        fields = tlbm.get_numpy_fields()

        # Update temperature
        im1.set_array(fields['T'].T)

        # Update velocity vectors
        quiver.set_UVC(fields['vx'][::4, ::4].T,
                      fields['vy'][::4, ::4].T)

        # Update vorticity (pre-computed in MLX)
        im2.set_array(fields.get('vorticity', np.zeros_like(fields['vx'])).T)

        # Update resource RGB visualization
        rgb_resources = np.zeros((fields['P_A'].T.shape[0], fields['P_A'].T.shape[1], 3))
        rgb_resources[:, :, 0] = np.clip(fields['P_A'].T, 0, 1)  # Red channel for A
        rgb_resources[:, :, 2] = np.clip(fields['P_B'].T, 0, 1)  # Blue channel for B
        im3.set_array(rgb_resources)

        # Update cell visualizations
        cell_pos = cells.get_cell_positions_numpy()
        if len(cell_pos) > 0:
            M_values, A_values, B_values, E_values = cells.get_cell_values_numpy()

            # Update scatter plots
            scatter1.set_offsets(cell_pos)
            scatter1.set_sizes(M_values * 50)

            # Update cell colors based on A/B content
            cell_colors = np.zeros((len(cell_pos), 3))
            cell_colors[:, 0] = np.clip(A_values / (A_values + B_values + 1e-6), 0, 1)  # Red
            cell_colors[:, 2] = np.clip(B_values / (A_values + B_values + 1e-6), 0, 1)  # Blue
            scatter3.set_offsets(cell_pos)
            scatter3.set_sizes(E_values * 100)
            scatter3.set_facecolors(cell_colors)
        else:
            scatter1.set_offsets(np.empty((0, 2)))
            scatter3.set_offsets(np.empty((0, 2)))

        # Update statistics
        stats = cells.get_stats()
        nu = tlbm.calculate_nusselt()

        steps_history.append(current_step)
        n_cells_history.append(stats['n_cells'])
        avg_M_history.append(stats['avg_M'] if stats['n_cells'] > 0 else 0)
        nu_history.append(nu)
        
        # Log division readiness and resource saturation periodically
        if current_step % 1000 == 0:
            # Get max resource values to check saturation
            # Use mx.where to mask dead cells with -inf, then take max
            if stats['n_cells'] > 0:
                max_A = float(mx.max(mx.where(cells.alive > 0, cells.A, -float('inf'))))
                max_B = float(mx.max(mx.where(cells.alive > 0, cells.B, -float('inf'))))
                max_E = float(mx.max(mx.where(cells.alive > 0, cells.E, -float('inf'))))
            else:
                max_A = max_B = max_E = 0
            
            print(f"\nStep {current_step}: Cells={stats['n_cells']:.0f}, "
                  f"Above threshold={stats['n_above_threshold']:.0f}, "
                  f"On cooldown={stats['n_on_cooldown']:.0f}, "
                  f"Ready to divide={stats['n_ready_to_divide']:.0f}, "
                  f"Max M={stats['max_M']:.2f}")
            print(f"  Resource maxima: A={max_A:.2f}/{cells.config.max_A}, "
                  f"B={max_B:.2f}/{cells.config.max_B}, "
                  f"E={max_E:.2f}/{cells.config.max_E}")

        # Update plots
        line_cells.set_data(steps_history, n_cells_history)
        line_M.set_data(steps_history, avg_M_history)
        line_nu.set_data(steps_history, nu_history)

        # Adjust axes
        if len(n_cells_history) > 1:
            ax4.set_ylim(0, max(10, max(n_cells_history) * 1.1))
            ax4_twin1.set_ylim(0, max(1, max(avg_M_history) * 1.1))
            ax4_twin2.set_ylim(min(nu_history) * 0.9, max(nu_history) * 1.1)

        # Performance info
        elapsed = time.time() - start_time
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0

        # Update title
        fig.suptitle(f'Coupled TLBM-Cell System - Step {current_step}/{num_steps} '
                    f'({steps_per_sec:.1f} steps/s) - Cells: {stats["n_cells"]}',
                    fontsize=14)

        return [im1, im2, im3, quiver, scatter1, scatter3,
                line_cells, line_M, line_nu]

    # Create animation
    frames = num_steps // update_interval
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()

    return anim


def main():
    """Run the coupled TLBM-Cell simulation"""

    print("Coupled TLBM-Cell System Simulation")
    print("===================================")

    # Set MLX device
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
    print(f"Using device: {mx.default_device()}")

    # Use centralized configuration
    config = DEFAULT_CONFIG

    # Create systems
    print(f"\nInitializing TLBM ({config.nx}×{config.ny} grid)...")
    tlbm = TLBM_MLX(nx=config.nx, ny=config.ny, config=config.tlbm, debug=False)

    print("Initializing dense cell system...")
    cells = DenseCellSystem(nx=config.nx, ny=config.ny, config=config.cells)

    # Seed initial cells randomly in bottom half
    print("Seeding initial cells...")

    # Number of initial cells
    n_initial_cells = 20

    # Generate random positions in bottom half of grid using MLX
    mx.random.seed(42)  # For reproducibility
    x_positions = mx.random.randint(5, config.nx - 5, shape=(n_initial_cells,))
    y_positions = mx.random.randint(5, config.ny // 2, shape=(n_initial_cells,))
    
    # Convert to list of tuples and remove duplicates
    initial_positions = list(set(zip(x_positions.tolist(), y_positions.tolist())))

    cells.seed_cells(initial_positions, M=0.5, A=0.3, B=0.3)

    print(f"Initial cells: {cells.n_cells}")
    print(f"\nResource configuration:")
    print(f"- Left source: {config.tlbm.resource_source_left}")
    print(f"- Right source: {config.tlbm.resource_source_right}")
    print("\nStarting simulation...")
    print("Expected behavior:")
    print("- Cells use A + E -> 2E for energy amplification")
    print("- Cells use B + E -> M for biomass production")
    print("- Resources A (red) and B (blue) diffuse from bottom")
    print("- Cells provide partial flow resistance")
    print()

    # Run visualization
    create_coupled_visualization(tlbm, cells, config, num_steps=50000, update_interval=config.update_interval)


if __name__ == "__main__":
    main()
