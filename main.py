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
    M_values, _ = cells.get_cell_values_numpy()
    if len(cell_pos) > 0:
        scatter1 = ax1.scatter(cell_pos[:, 0], cell_pos[:, 1],
                             s=M_values * 50, c='blue',
                             alpha=0.6, edgecolors='darkblue')
    else:
        scatter1 = ax1.scatter([], [], s=[], c='blue')

    # 2. Vorticity field
    vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
    im2 = ax2.imshow(vort.T, cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
    ax2.set_title('Vorticity')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)

    # 3. Resource concentration with cells
    im3 = ax3.imshow(fields['P'].T, cmap='Greens', origin='lower',
                     vmin=0, vmax=1, alpha=0.8)
    ax3.set_title('Resource Concentration & Cells')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Resource (P)')

    # Cell scatter for resource plot (size based on internal R)
    if len(cell_pos) > 0:
        _, R_values = cells.get_cell_values_numpy()
        scatter3 = ax3.scatter(cell_pos[:, 0], cell_pos[:, 1],
                             s=R_values * 100, c='darkgreen',
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
                 f"Cells: reaction_rate={cells.config.reaction_rate:.2f}, "
                 f"division_threshold={cells.config.division_threshold:.1f}")
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
                # Update cells with resource and temperature fields
                P_field = tlbm.P[tlbm.phys_x, tlbm.phys_y]
                T_field = tlbm.T[tlbm.phys_x, tlbm.phys_y]
                cells.step(P_field, T_field)

                # Apply cell effects to TLBM
                # 1. Update cell presence for flow resistance
                tlbm.cell_presence = cells.cell_presence

                # 2. Add heat from metabolism to temperature
                if mx.any(cells.heat_generation > 0):
                    T_phys = tlbm.T[tlbm.phys_x, tlbm.phys_y]
                    T_new = T_phys + cells.heat_generation * config.coupling.heat_addition_factor
                    tlbm.T[tlbm.phys_x, tlbm.phys_y] = T_new

                    # Update temperature distributions
                    for k in range(9):
                        tlbm.g[tlbm.phys_x, tlbm.phys_y, k] = (
                            tlbm.w[k] * tlbm.T[tlbm.phys_x, tlbm.phys_y]
                        )

                # 3. Extract resources
                if mx.any(cells.resource_extraction > 0):
                    P_phys = tlbm.P[tlbm.phys_x, tlbm.phys_y]
                    P_new = mx.maximum(0, P_phys - cells.resource_extraction * config.coupling.resource_extraction_factor)
                    tlbm.P[tlbm.phys_x, tlbm.phys_y] = P_new

                    # Update resource distributions
                    for k in range(9):
                        tlbm.r[tlbm.phys_x, tlbm.phys_y, k] = (
                            tlbm.w[k] * tlbm.P[tlbm.phys_x, tlbm.phys_y]
                        )

        current_step = frame * update_interval

        # Get updated fields
        fields = tlbm.get_numpy_fields()

        # Update temperature
        im1.set_array(fields['T'].T)

        # Update velocity vectors
        quiver.set_UVC(fields['vx'][::4, ::4].T,
                      fields['vy'][::4, ::4].T)

        # Update vorticity
        vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
        im2.set_array(vort.T)

        # Update resource
        im3.set_array(fields['P'].T)

        # Update cell visualizations
        cell_pos = cells.get_cell_positions_numpy()
        if len(cell_pos) > 0:
            M_values, R_values = cells.get_cell_values_numpy()

            # Update scatter plots
            scatter1.set_offsets(cell_pos)
            scatter1.set_sizes(M_values * 50)

            scatter3.set_offsets(cell_pos)
            scatter3.set_sizes(R_values * 100)
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
        
        # Log division readiness periodically
        if current_step % 1000 == 0:
            print(f"\nStep {current_step}: Cells={stats['n_cells']:.0f}, "
                  f"Above threshold={stats['n_above_threshold']:.0f}, "
                  f"On cooldown={stats['n_on_cooldown']:.0f}, "
                  f"Ready to divide={stats['n_ready_to_divide']:.0f}, "
                  f"Max M={stats['max_M']:.2f}")

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
    import random

    # Number of initial cells
    n_initial_cells = 20

    # Generate random positions in bottom half of grid
    initial_positions = []
    for _ in range(n_initial_cells):
        x = random.randint(5, config.nx - 5)  # Keep away from edges
        y = random.randint(5, config.ny // 2)  # Bottom half only
        initial_positions.append((x, y))

    # Remove duplicates
    initial_positions = list(set(initial_positions))

    cells.seed_cells(initial_positions, M=0.5, R=0.3)

    print(f"Initial cells: {cells.n_cells}")
    print("\nStarting simulation...")
    print("Expected behavior:")
    print("- Cells will consume resources and divide")
    print("- Metabolism generates heat, affecting convection")
    print("- Cells provide partial flow resistance")
    print("- Resource gradients drive cell movement patterns")
    print()

    # Run visualization
    create_coupled_visualization(tlbm, cells, config, num_steps=50000, update_interval=config.update_interval)


if __name__ == "__main__":
    main()
