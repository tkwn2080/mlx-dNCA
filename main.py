"""
MLX TLBM with Vispy visualization for better real-time performance
"""

import mlx.core as mx
import numpy as np
import vispy
from vispy import scene, app
from vispy.color import get_colormap
import time
from tlbm import TLBM_MLX, TLBMConfig


class TLBMVisualizer:
    """Real-time visualization for TLBM using Vispy"""

    def __init__(self, tlbm: TLBM_MLX):
        self.tlbm = tlbm
        self.step_count = 0
        self.start_time = time.time()

        # Create canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',
                                       size=(1200, 800), title='MLX TLBM Simulation')
        self.canvas.show()

        # Create viewboxes
        grid = self.canvas.central_widget.add_grid()

        # Get initial fields to set up images
        fields = self.tlbm.get_numpy_fields()

        # Temperature view
        self.temp_view = grid.add_view(row=0, col=0, border_color='black')
        self.temp_image = scene.visuals.Image(fields['T'].T, parent=self.temp_view.scene,
                                             cmap='hot', clim=(0, 1))
        self.temp_view.camera = scene.PanZoomCamera(aspect=1)
        self.temp_view.camera.set_range()
        scene.visuals.Text('Temperature', parent=self.temp_view.scene,
                          pos=(10, 10), color='white', font_size=14)

        # Vorticity view
        vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
        self.vort_view = grid.add_view(row=0, col=1, border_color='black')
        self.vort_image = scene.visuals.Image(vort.T, parent=self.vort_view.scene,
                                             cmap='RdBu', clim=(-0.1, 0.1))
        self.vort_view.camera = scene.PanZoomCamera(aspect=1)
        self.vort_view.camera.set_range()
        scene.visuals.Text('Vorticity', parent=self.vort_view.scene,
                          pos=(10, 10), color='black', font_size=14)

        # Resource view
        self.res_view = grid.add_view(row=1, col=0, border_color='black')
        self.res_image = scene.visuals.Image(fields['P'].T, parent=self.res_view.scene,
                                            cmap='viridis', clim=(0, 1))
        self.res_view.camera = scene.PanZoomCamera(aspect=1)
        self.res_view.camera.set_range()
        scene.visuals.Text('Resources', parent=self.res_view.scene,
                          pos=(10, 10), color='white', font_size=14)

        # Info view
        self.info_view = grid.add_view(row=1, col=1, border_color='black')
        self.info_text = scene.visuals.Text('', parent=self.info_view.scene,
                                           pos=(10, 50), color='black', font_size=12)

        # Nusselt plot
        self.nu_history = []
        self.step_history = []
        self.nu_line = scene.visuals.Line(parent=self.info_view.scene,
                                         color='blue', width=2)

        # Timer for updates
        self.timer = app.Timer(interval=0.05, connect=self.update, start=True)

    def update(self, event):
        """Update visualization"""
        # Run simulation steps
        steps_per_update = 50
        for _ in range(steps_per_update):
            if not self.tlbm.step():
                # Simulation failed - stop timer
                self.timer.stop()
                print("\nSimulation stopped due to numerical instability")
                return
            self.step_count += 1

        # Get fields
        fields = self.tlbm.get_numpy_fields()

        # Update temperature
        self.temp_image.set_data(fields['T'].T)

        # Calculate and update vorticity
        vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)
        self.vort_image.set_data(vort.T)

        # Update resources
        self.res_image.set_data(fields['P'].T)

        # Calculate Nusselt number
        nu = self.tlbm.calculate_nusselt()
        self.nu_history.append(nu)
        self.step_history.append(self.step_count)

        # Update info text
        elapsed = time.time() - self.start_time
        steps_per_sec = self.step_count / elapsed if elapsed > 0 else 0

        info_str = f"Steps: {self.step_count}\n"
        info_str += f"Speed: {steps_per_sec:.1f} steps/s\n"
        info_str += f"Nu: {nu:.3f}\n"
        info_str += f"Ra: {self.tlbm.config.Ra:.1e}\n"
        info_str += f"Pr: {self.tlbm.config.Pr:.2f}"

        self.info_text.text = info_str

        # Update Nusselt plot (show last 1000 points)
        if len(self.nu_history) > 2:
            n_show = min(1000, len(self.nu_history))
            x = np.array(self.step_history[-n_show:])
            y = np.array(self.nu_history[-n_show:])

            # Normalize for display
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6) * 300 + 50
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6) * 200 + 250

            pos = np.column_stack([x_norm, y_norm])
            self.nu_line.set_data(pos=pos)

        # Stop after 10000 steps
        if self.step_count >= 10000:
            self.timer.stop()
            print(f"\nSimulation complete!")
            print(f"Final Nusselt number: {nu:.3f}")
            print(f"Average speed: {steps_per_sec:.1f} steps/s")

    def run(self):
        """Start the visualization"""
        app.run()


def run_mlx_tlbm_vispy():
    """Run MLX TLBM with Vispy visualization"""

    print("MLX TLBM - Vispy Visualization")
    print("==============================")

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
    print()

    # Create TLBM instance - using smaller grid to match main.py
    print("Initializing TLBM...")
    tlbm = TLBM_MLX(nx=128, ny=64, config=config)

    # Check device
    print(f"Using device: {mx.default_device()}")
    print()

    # Create and run visualizer
    print("Starting visualization...")
    print("Controls: Mouse wheel to zoom, drag to pan")
    viz = TLBMVisualizer(tlbm)
    viz.run()


if __name__ == "__main__":
    # Set default device
    mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)

    # Run visualization
    run_mlx_tlbm_vispy()
