"""
Dense grid MLX-based cell system for TLBM coupling
Uses full grid arrays for all properties, enabling fully vectorized operations
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, Dict, Optional
from .config import CellConfig


class DenseCellSystem:
    """Dense grid cellular system with autocatalytic reactions"""
    
    def __init__(self, nx: int, ny: int, config: CellConfig):
        self.nx = nx
        self.ny = ny
        self.config = config
        
        # Dense grids for cell properties
        self.M = mx.zeros((nx, ny), dtype=mx.float32)        # Metabolite levels
        self.R = mx.zeros((nx, ny), dtype=mx.float32)        # Resource levels
        self.alive = mx.zeros((nx, ny), dtype=mx.float32)    # 1.0 if alive, 0.0 if dead
        
        # Output fields for TLBM coupling
        self.heat_generation = mx.zeros((nx, ny), dtype=mx.float32)
        self.resource_extraction = mx.zeros((nx, ny), dtype=mx.float32)
        
        # Moore neighborhood kernel for convolution
        self.neighbor_kernel = mx.array([[1, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1]], dtype=mx.float32)
        
        # Diffusion kernel for resource sharing (normalized)
        # Using 8-neighbor (Moore) kernel for isotropic diffusion
        # Corner neighbors weighted by 1/sqrt(2) for equal distance weighting
        corner_weight = 1.0 / mx.sqrt(2.0)
        self.diffusion_kernel = mx.array([[corner_weight, 1, corner_weight],
                                         [1, -4 - 4*corner_weight, 1],
                                         [corner_weight, 1, corner_weight]], dtype=mx.float32)
        # Normalize so sum = 0 and weights sum to 1
        self.diffusion_kernel = self.diffusion_kernel / (4 + 4 * corner_weight)
        
        # Directional kernels for division (8 directions)
        # Each kernel represents division in one direction
        self.division_kernels = [
            mx.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),  # North
            mx.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),  # East
            mx.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),  # South
            mx.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),  # West
            mx.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),  # NE
            mx.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # SE
            mx.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),  # SW
            mx.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # NW
        ]
        self.division_kernels = [k.astype(mx.float32) for k in self.division_kernels]
        
        # Division probability decay (cells can't divide into recently divided spots)
        self.division_cooldown = mx.zeros((nx, ny), dtype=mx.float32)
        
        # Random direction selection for each cell
        self.division_direction = mx.zeros((nx, ny), dtype=mx.int32)
        
    def seed_cells(self, positions: list, M: float = 1.0, R: float = 0.5):
        """Seed initial cells at given positions"""
        for x, y in positions:
            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.M[x, y] = M
                self.R[x, y] = R
                self.alive[x, y] = 1.0
                
    def compute_resource_gradients(self, P_field: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute resource gradients using finite differences"""
        # Pad for boundary handling
        P_padded = mx.pad(P_field, ((1, 1), (1, 1)), mode='edge')
        
        # Central differences
        grad_x = (P_padded[2:, 1:-1] - P_padded[:-2, 1:-1]) / 2.0
        grad_y = (P_padded[1:-1, 2:] - P_padded[1:-1, :-2]) / 2.0
        
        return grad_x, grad_y
        
    def count_neighbors(self) -> mx.array:
        """Count living neighbors for each cell using convolution"""
        # MLX uses NHWC format: (batch, height, width, channels)
        alive_4d = self.alive.reshape(1, self.nx, self.ny, 1)
        # Weight format: (out_channels, kernel_h, kernel_w, in_channels)
        kernel_4d = self.neighbor_kernel.reshape(1, 3, 3, 1)
        
        # Convolve with padding to handle boundaries
        neighbor_count = mx.conv2d(alive_4d, kernel_4d, padding=1)
        
        # Reshape back to 2D (remove batch and channel dimensions)
        return neighbor_count.squeeze()
        
    def step(self, P_field: mx.array, T_field: mx.array = None):
        """Perform one cell update step using dense operations"""
        
        # Generate stochastic update mask (currently unused - kept for potential future use)
        update_mask = mx.random.uniform(shape=(self.nx, self.ny)) < 0.5
        
        # 1. Resource ingestion based on concentration difference (deterministic - passive diffusion)
        ingestion = self.config.ingestion_rate * mx.maximum(0, P_field - self.R)
        ingestion = ingestion * self.alive  # Only living cells, but deterministic
        
        # 2. Autocatalytic reaction: M + R -> 2M (deterministic - core metabolism)
        # Temperature-dependent reaction rate (Q10 ~ 2, i.e., doubles every 10°C)
        if T_field is not None:
            # Assuming T_field is normalized 0-1, where 0.5 is "normal" temperature
            # Using exponential relationship: rate = base_rate * exp(k * (T - T_ref))
            # For Q10=2: k = ln(2)/10 ≈ 0.0693 per 0.1 normalized units
            temp_factor = mx.exp(0.693 * (T_field - 0.5))  # Q10 = 2
            effective_rate = self.config.reaction_rate * temp_factor
        else:
            effective_rate = self.config.reaction_rate
            
        reaction_amount = effective_rate * self.M * self.R * self.alive  # No stochastic mask
        
        # 3. Maintenance cost (ALWAYS happens - no stochastic mask)
        # Temperature-dependent maintenance: increases above critical temperature
        if T_field is not None:
            # Calculate heat stress factor
            # Below critical temp: factor = 1
            # Above critical temp: factor increases linearly up to heat_stress_factor
            heat_stress = mx.maximum(0, (T_field - self.config.critical_temp) / (1.0 - self.config.critical_temp))
            maintenance_factor = 1.0 + heat_stress * (self.config.heat_stress_factor - 1.0)
        else:
            maintenance_factor = 1.0
            
        maintenance_cost = self.config.maintenance_rate * maintenance_factor * self.alive
        
        # 4. Update internal states
        self.R = self.R + ingestion - reaction_amount
        self.M = self.M + reaction_amount - maintenance_cost
        
        # 4b. Resource sharing between adjacent cells (passive diffusion)
        # Only share R between living cells
        R_alive = self.R * self.alive
        
        # Compute diffusion using convolution
        R_4d = R_alive.reshape(1, self.nx, self.ny, 1)  # NHWC format
        diffusion_kernel_4d = self.diffusion_kernel.reshape(1, 3, 3, 1)
        
        # Apply diffusion (Laplacian)
        R_diffusion = mx.conv2d(R_4d, diffusion_kernel_4d, padding=1).squeeze()
        
        # Apply diffusion with a rate, only to living cells
        self.R = self.R + self.config.diffusion_rate * R_diffusion * self.alive
        
        # Ensure non-negative
        self.R = mx.maximum(0, self.R)
        
        # 5. Generate heat from metabolism
        reaction_heat = self.config.heat_per_reaction * reaction_amount
        
        # Non-linear heat generation from maintenance
        # At normal maintenance: normal heat
        # At high maintenance (heat stress): disproportionately more heat (positive feedback)
        if T_field is not None:
            # Heat generation increases quadratically with maintenance factor
            heat_scaling = maintenance_factor ** 1.5  # Non-linear scaling
            maintenance_heat = self.config.heat_per_maintenance * maintenance_cost * heat_scaling
        else:
            maintenance_heat = self.config.heat_per_maintenance * maintenance_cost
            
        self.heat_generation = reaction_heat + maintenance_heat
        
        # 6. Resource extraction (rate per step)
        self.resource_extraction = ingestion
        
        # 7. Handle cell death
        self.alive = mx.where((self.M >= self.config.death_threshold) & (self.M > 0), 
                             self.alive, 0.0)
        
        # Dead cells lose their contents
        self.M = self.M * self.alive
        self.R = self.R * self.alive
        
        # 8. Handle cell division
        self._handle_division()
        
        # 9. Update division cooldown (decrements by 1 each step)
        self.division_cooldown = mx.maximum(0, self.division_cooldown - 1)
        
        # Ensure arrays are evaluated
        mx.eval(self.alive, self.heat_generation, self.resource_extraction)
        
    def _handle_division(self):
        """Handle cell division with conflict detection for true binary fission"""
        # Find cells ready to divide
        can_divide = (self.M > self.config.division_threshold) & (self.alive > 0)
        
        # Count free neighbors
        neighbor_count = self.count_neighbors()
        free_neighbors = 8 - neighbor_count
        
        # Cells can only divide if they have free neighbors and cooldown is 0
        will_divide = can_divide & (free_neighbors > 0) & (self.division_cooldown == 0)
        
        # Debug: Log division readiness
        n_above_threshold = int(mx.sum(can_divide))
        n_cooldown_ok = int(mx.sum(can_divide & (self.division_cooldown == 0)))
        n_has_space = int(mx.sum(will_divide))
        
        if not mx.any(will_divide):
            if n_above_threshold > 0:
                print(f"Division blocked: {n_above_threshold} above threshold, "
                      f"{n_cooldown_ok} cooldown OK, {n_has_space} have space")
            return
            
        # Apply stochastic division mask (30% chance to divide even when ready)
        division_mask = mx.random.uniform(shape=(self.nx, self.ny)) < self.config.division_probability
        will_divide = will_divide & division_mask
        
        n_will_divide = int(mx.sum(will_divide))
        if not mx.any(will_divide):
            print(f"Division blocked by stochastic mask: {n_has_space} ready, 0 selected")
            return
            
        print(f"Division occurring: {n_will_divide} cells dividing")
            
        # For each dividing cell, randomly select a division direction
        random_dirs = mx.random.randint(0, 8, shape=(self.nx, self.ny))
        
        # Phase 1: Detect conflicts - count how many parents want each position
        conflict_map = mx.zeros((self.nx, self.ny), dtype=mx.float32)
        
        for dir_idx in range(8):
            # Find cells dividing in this direction
            dividing_in_dir = (will_divide & (random_dirs == dir_idx)).astype(mx.float32)
            
            if mx.any(dividing_in_dir):
                # Convert to 4D for convolution
                dividing_4d = dividing_in_dir.reshape(1, self.nx, self.ny, 1)
                kernel_4d = self.division_kernels[dir_idx].reshape(1, 3, 3, 1)
                
                # Add to conflict map
                claims = mx.conv2d(dividing_4d, kernel_4d, padding=1).squeeze()
                conflict_map = conflict_map + claims
        
        # Identify conflict-free and conflicted positions
        no_conflict = (conflict_map == 1) & (self.alive == 0) & (self.division_cooldown == 0)
        has_conflict = (conflict_map > 1) & (self.alive == 0) & (self.division_cooldown == 0)
        
        # Store which parents have successfully divided
        parent_divided = mx.zeros((self.nx, self.ny), dtype=mx.bool_)
        
        # Phase 2: Process conflict-free divisions (fully vectorized)
        for dir_idx in range(8):
            dividing_in_dir = will_divide & (random_dirs == dir_idx) & (~parent_divided)
            
            if not mx.any(dividing_in_dir):
                continue
                
            # Convert to 4D
            dividing_4d = dividing_in_dir.reshape(1, self.nx, self.ny, 1).astype(mx.float32)
            kernel_4d = self.division_kernels[dir_idx].reshape(1, 3, 3, 1)
            
            # Find offspring positions
            offspring_pos = mx.conv2d(dividing_4d, kernel_4d, padding=1).squeeze() > 0
            
            # Only process if offspring position is conflict-free
            valid_offspring = offspring_pos & no_conflict
            
            if mx.any(valid_offspring):
                # Get parent resources
                parent_M_4d = (self.M * dividing_in_dir).reshape(1, self.nx, self.ny, 1)
                parent_R_4d = (self.R * dividing_in_dir).reshape(1, self.nx, self.ny, 1)
                
                # Calculate offspring resources
                m_offspring = mx.conv2d(parent_M_4d, kernel_4d, padding=1).squeeze()
                r_offspring = mx.conv2d(parent_R_4d, kernel_4d, padding=1).squeeze()
                
                # Place offspring (50% of parent's resources)
                self.M = mx.where(valid_offspring, m_offspring * 0.5, self.M)
                self.R = mx.where(valid_offspring, r_offspring * 0.5, self.R)
                self.alive = mx.where(valid_offspring, 1.0, self.alive)
                self.division_cooldown = mx.where(valid_offspring, self.config.offspring_cooldown, self.division_cooldown)
                
                # Find which parents successfully divided (reverse convolution)
                success_4d = valid_offspring.reshape(1, self.nx, self.ny, 1).astype(mx.float32)
                reverse_kernel = self.division_kernels[7-dir_idx].reshape(1, 3, 3, 1)
                parent_success = mx.conv2d(success_4d, reverse_kernel, padding=1).squeeze() > 0
                
                # Update parent resources and mark as divided (keep 50%)
                successful_parents = dividing_in_dir & parent_success
                self.M = mx.where(successful_parents, self.M * 0.5, self.M)
                self.R = mx.where(successful_parents, self.R * 0.5, self.R)
                self.division_cooldown = mx.where(successful_parents, self.config.division_cooldown, self.division_cooldown)
                parent_divided = parent_divided | successful_parents
        
        # Phase 3: Handle conflicts (minimal sequential processing)
        if mx.any(has_conflict):
            # Convert to numpy for conflict resolution
            conflict_positions = np.argwhere(np.array(has_conflict, copy=False))
            
            for cx, cy in conflict_positions:
                # Find all parents trying to divide into this position
                potential_parents = []
                
                for dx, dy in [(0, 1), (-1, 0), (0, -1), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    px, py = cx - dx, cy - dy
                    if 0 <= px < self.nx and 0 <= py < self.ny:
                        # Convert to numpy arrays for indexing
                        will_divide_np = np.array(will_divide, copy=False)
                        parent_divided_np = np.array(parent_divided, copy=False)
                        random_dirs_np = np.array(random_dirs, copy=False)
                        
                        if will_divide_np[px, py] and not parent_divided_np[px, py]:
                            # Check if this parent wants to divide toward (cx, cy)
                            dir_map = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3,
                                      (1, -1): 4, (1, 1): 5, (-1, 1): 6, (-1, -1): 7}
                            if (dx, dy) in dir_map and random_dirs_np[px, py] == dir_map[(dx, dy)]:
                                potential_parents.append((px, py))
                
                # Randomly select one parent
                if potential_parents:
                    import random
                    px, py = random.choice(potential_parents)
                    
                    # Convert indices to Python int for MLX indexing
                    cx_int, cy_int = int(cx), int(cy)
                    px_int, py_int = int(px), int(py)
                    
                    # Perform division (50/50 split)
                    self.M[cx_int, cy_int] = float(self.M[px_int, py_int]) * 0.5
                    self.R[cx_int, cy_int] = float(self.R[px_int, py_int]) * 0.5
                    self.alive[cx_int, cy_int] = 1.0
                    self.division_cooldown[cx_int, cy_int] = self.config.offspring_cooldown
                    
                    # Update parent (keep 50%)
                    self.M[px_int, py_int] = float(self.M[px_int, py_int]) * 0.5
                    self.R[px_int, py_int] = float(self.R[px_int, py_int]) * 0.5
                    self.division_cooldown[px_int, py_int] = self.config.division_cooldown
                    # Update parent_divided - create a boolean mask and OR it
                    mask = mx.zeros((self.nx, self.ny), dtype=mx.bool_)
                    mask[px_int, py_int] = True
                    parent_divided = parent_divided | mask
        
        
    def get_stats(self) -> Dict[str, float]:
        """Get system statistics"""
        n_cells = float(mx.sum(self.alive))
        
        if n_cells == 0:
            return {
                'n_cells': 0,
                'total_M': 0.0,
                'total_R': 0.0,
                'avg_M': 0.0,
                'avg_R': 0.0,
                'n_above_threshold': 0,
                'n_on_cooldown': 0,
                'max_M': 0.0,
                'n_ready_to_divide': 0
            }
            
        total_M = float(mx.sum(self.M))
        total_R = float(mx.sum(self.R))
        
        # Division readiness stats
        above_threshold = (self.M > self.config.division_threshold) & (self.alive > 0)
        on_cooldown = (self.division_cooldown > 0) & (self.alive > 0)
        ready_no_cooldown = above_threshold & (~on_cooldown)
        
        # Check space availability
        neighbor_count = self.count_neighbors()
        has_space = (neighbor_count < 8) & (self.alive > 0)
        ready_to_divide = ready_no_cooldown & has_space
        
        return {
            'n_cells': n_cells,
            'total_M': total_M,
            'total_R': total_R,
            'avg_M': total_M / n_cells,
            'avg_R': total_R / n_cells,
            'n_above_threshold': float(mx.sum(above_threshold)),
            'n_on_cooldown': float(mx.sum(on_cooldown)),
            'max_M': float(mx.max(self.M)),
            'n_ready_to_divide': float(mx.sum(ready_to_divide))
        }
        
    def get_cell_positions_numpy(self) -> np.ndarray:
        """Get positions of living cells for visualization"""
        alive_np = np.array(self.alive)
        positions = np.argwhere(alive_np > 0)
        return positions
        
    def get_cell_values_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get M and R values for living cells"""
        alive_np = np.array(self.alive)
        M_np = np.array(self.M)
        R_np = np.array(self.R)
        
        # Get values only for living cells
        alive_mask = alive_np > 0
        M_values = M_np[alive_mask]
        R_values = R_np[alive_mask]
        
        return M_values, R_values
        
    @property
    def cell_presence(self):
        """For compatibility with TLBM interface"""
        return self.alive
        
    @property 
    def n_cells(self):
        """For compatibility with sparse interface"""
        return int(mx.sum(self.alive))