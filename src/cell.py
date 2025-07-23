"""
Dense grid MLX-based cell system for TLBM coupling
Uses full grid arrays for all properties, enabling fully vectorized operations
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, Dict, Optional
from functools import partial
from .config import CellConfig


class DenseCellSystem:
    """Dense grid cellular system with autocatalytic reactions"""
    
    def __init__(self, nx: int, ny: int, config: CellConfig):
        self.nx = nx
        self.ny = ny
        self.config = config
        
        # Dense grids for cell properties
        self.M = mx.zeros((nx, ny), dtype=mx.float32)        # Metabolite levels
        self.A = mx.zeros((nx, ny), dtype=mx.float32)        # Resource A levels
        self.B = mx.zeros((nx, ny), dtype=mx.float32)        # Resource B levels
        self.E = mx.zeros((nx, ny), dtype=mx.float32)        # Energy levels
        self.alive = mx.zeros((nx, ny), dtype=mx.float32)    # 1.0 if alive, 0.0 if dead
        
        # Output fields for TLBM coupling
        self.resource_A_extraction = mx.zeros((nx, ny), dtype=mx.float32)
        self.resource_B_extraction = mx.zeros((nx, ny), dtype=mx.float32)
        
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
        
    def seed_cells(self, positions: list, M: float = 1.0, A: float = 0.3, B: float = 0.3):
        """Seed initial cells at given positions"""
        for x, y in positions:
            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.M[x, y] = M
                self.A[x, y] = A
                self.B[x, y] = B
                self.E[x, y] = self.config.initial_E
                self.alive[x, y] = 1.0
                
        
    def _compute_reactions(self, A, B, E, alive):
        """Compute metabolic reactions"""
        # Step 1: A + E -> 2E (energy amplification)
        reaction_1_amount = self.config.reaction_rate_1 * A * E * alive
        
        # Step 2: B + E -> M (biomass production)
        reaction_2_amount = self.config.reaction_rate_2 * B * E * alive
        
        return reaction_1_amount, reaction_2_amount
    
    def _apply_diffusion(self, A_alive, B_alive, alive, diffusion_kernel):
        """Apply resource diffusion between cells"""
        # Stack A and B into a multi-channel array for batched convolution
        # Shape: (1, nx, ny, 2) - batch=1, height=nx, width=ny, channels=2
        AB_stacked = mx.stack([A_alive, B_alive], axis=-1).reshape(1, self.nx, self.ny, 2)
        
        # Prepare kernel for multi-channel convolution
        # Original kernel shape: (3, 3)
        # Need shape: (out_channels=2, kernel_h=3, kernel_w=3, in_channels=2)
        # We want each channel processed independently, so create diagonal filter
        kernel_2ch = mx.zeros((2, 3, 3, 2))
        kernel_2ch[0, :, :, 0] = diffusion_kernel  # A channel
        kernel_2ch[1, :, :, 1] = diffusion_kernel  # B channel
        
        # Apply batched convolution
        AB_diffusion = mx.conv2d(AB_stacked, kernel_2ch, padding=1)
        
        # Unstack and apply alive mask
        A_diffusion = AB_diffusion[0, :, :, 0] * alive
        B_diffusion = AB_diffusion[0, :, :, 1] * alive
        
        return A_diffusion, B_diffusion
        
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
        
    def step(self, P_A_field: mx.array, P_B_field: mx.array):
        """Perform one cell update step using dense operations"""
        
        # Generate stochastic update mask (currently unused - kept for potential future use)
        update_mask = mx.random.uniform(shape=(self.nx, self.ny)) < 0.5
        
        # 1. Resource ingestion with saturation limits (deterministic - passive diffusion)
        # Calculate headroom (how much more we can take before saturation)
        headroom_A = mx.maximum(0, self.config.max_A - self.A)
        headroom_B = mx.maximum(0, self.config.max_B - self.B)
        
        # Gradient-driven uptake
        gradient_A = mx.maximum(0, P_A_field - self.A)
        gradient_B = mx.maximum(0, P_B_field - self.B)
        
        # Take minimum of gradient desire and headroom
        ingestion_A = self.config.ingestion_rate * mx.minimum(gradient_A, headroom_A) * self.alive
        ingestion_B = self.config.ingestion_rate * mx.minimum(gradient_B, headroom_B) * self.alive
        
        # 2. Metabolic reactions (using compiled function)
        reaction_1_amount, reaction_2_amount = self._compute_reactions(
            self.A, self.B, self.E, self.alive
        )
        
        # 3. Maintenance cost (simple, no temperature dependence)
        maintenance_cost = self.config.maintenance_rate * self.alive
        
        # 4. Update internal states
        self.A = self.A + ingestion_A - reaction_1_amount
        self.B = self.B + ingestion_B - reaction_2_amount
        self.E = self.E + reaction_1_amount - reaction_2_amount  # Gain from reaction 1, consumed in reaction 2
        self.M = self.M + reaction_2_amount - maintenance_cost
        
        # Apply saturation limits
        self.E = mx.minimum(self.E, self.config.max_E)
        
        # 4b. Resource sharing between adjacent cells (passive diffusion)
        # Share both A and B between living cells
        A_alive = self.A * self.alive
        B_alive = self.B * self.alive
        
        # Compute diffusion using compiled function with batched convolution
        A_diffusion, B_diffusion = self._apply_diffusion(
            A_alive, B_alive, self.alive, self.diffusion_kernel
        )
        
        # Apply diffusion with rate
        self.A = self.A + self.config.diffusion_rate * A_diffusion
        self.B = self.B + self.config.diffusion_rate * B_diffusion
        
        # Ensure non-negative
        self.A = mx.maximum(0, self.A)
        self.B = mx.maximum(0, self.B)
        self.E = mx.maximum(0, self.E)
        
        # 5. Resource extraction (rate per step)
        self.resource_A_extraction = ingestion_A
        self.resource_B_extraction = ingestion_B
        
        # 6. Handle cell death (from M or E depletion)
        self.alive = mx.where((self.M >= self.config.death_threshold_M) & 
                             (self.E >= self.config.death_threshold_E), 
                             self.alive, 0.0)
        
        # Dead cells lose their contents
        self.M = self.M * self.alive
        self.A = self.A * self.alive
        self.B = self.B * self.alive
        self.E = self.E * self.alive
        
        # 8. Handle cell division
        self._handle_division()
        
        # 9. Update division cooldown (decrements by 1 each step)
        self.division_cooldown = mx.maximum(0, self.division_cooldown - 1)
        
        # Ensure all arrays are evaluated to prevent computation buildup
        mx.eval(self.M, self.A, self.B, self.E, self.alive,
                self.resource_A_extraction, self.resource_B_extraction,
                self.division_cooldown)
        
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
                parent_A_4d = (self.A * dividing_in_dir).reshape(1, self.nx, self.ny, 1)
                parent_B_4d = (self.B * dividing_in_dir).reshape(1, self.nx, self.ny, 1)
                parent_E_4d = (self.E * dividing_in_dir).reshape(1, self.nx, self.ny, 1)
                
                # Calculate offspring resources
                m_offspring = mx.conv2d(parent_M_4d, kernel_4d, padding=1).squeeze()
                a_offspring = mx.conv2d(parent_A_4d, kernel_4d, padding=1).squeeze()
                b_offspring = mx.conv2d(parent_B_4d, kernel_4d, padding=1).squeeze()
                e_offspring = mx.conv2d(parent_E_4d, kernel_4d, padding=1).squeeze()
                
                # Place offspring (50% of parent's resources)
                self.M = mx.where(valid_offspring, m_offspring * 0.5, self.M)
                self.A = mx.where(valid_offspring, a_offspring * 0.5, self.A)
                self.B = mx.where(valid_offspring, b_offspring * 0.5, self.B)
                self.E = mx.where(valid_offspring, e_offspring * 0.5, self.E)
                self.alive = mx.where(valid_offspring, 1.0, self.alive)
                self.division_cooldown = mx.where(valid_offspring, self.config.offspring_cooldown, self.division_cooldown)
                
                # Find which parents successfully divided (reverse convolution)
                success_4d = valid_offspring.reshape(1, self.nx, self.ny, 1).astype(mx.float32)
                reverse_kernel = self.division_kernels[7-dir_idx].reshape(1, 3, 3, 1)
                parent_success = mx.conv2d(success_4d, reverse_kernel, padding=1).squeeze() > 0
                
                # Update parent resources and mark as divided (keep 50%)
                successful_parents = dividing_in_dir & parent_success
                self.M = mx.where(successful_parents, self.M * 0.5, self.M)
                self.A = mx.where(successful_parents, self.A * 0.5, self.A)
                self.B = mx.where(successful_parents, self.B * 0.5, self.B)
                self.E = mx.where(successful_parents, self.E * 0.5, self.E)
                self.division_cooldown = mx.where(successful_parents, self.config.division_cooldown, self.division_cooldown)
                parent_divided = parent_divided | successful_parents
        
        # Phase 3: Handle conflicts (using MLX operations)
        if mx.any(has_conflict):
            # For each conflicted position, we need to randomly select one parent
            # Since MLX doesn't have argwhere, we'll use a different approach
            
            # Create random priorities for each dividing cell
            random_priorities = mx.random.uniform(shape=(self.nx, self.ny))
            random_priorities = mx.where(will_divide & (~parent_divided), random_priorities, -1.0)
            
            # For each direction, check which parent has highest priority for each conflict position
            for dir_idx in range(8):
                dividing_in_dir = will_divide & (random_dirs == dir_idx) & (~parent_divided)
                
                if not mx.any(dividing_in_dir):
                    continue
                    
                # Get parent priorities for this direction
                parent_priority_4d = (random_priorities * dividing_in_dir).reshape(1, self.nx, self.ny, 1).astype(mx.float32)
                kernel_4d = self.division_kernels[dir_idx].reshape(1, 3, 3, 1)
                
                # Find max priority at each offspring position
                offspring_max_priority = mx.conv2d(parent_priority_4d, kernel_4d, padding=1).squeeze()
                
                # Check if this parent wins the conflict (has max priority)
                # Reverse convolution to find which parents match the max priority
                max_priority_4d = offspring_max_priority.reshape(1, self.nx, self.ny, 1).astype(mx.float32)
                reverse_kernel = self.division_kernels[7-dir_idx].reshape(1, 3, 3, 1)
                parent_matches_max = mx.conv2d(max_priority_4d, reverse_kernel, padding=1).squeeze()
                
                # Parent wins if their priority matches the max at offspring position
                winning_parents = dividing_in_dir & (mx.abs(random_priorities - parent_matches_max) < 1e-6)
                
                # Only process conflicts
                valid_conflicts = winning_parents & has_conflict
                
                if mx.any(valid_conflicts):
                    # Get parent resources
                    parent_M_4d = (self.M * valid_conflicts).reshape(1, self.nx, self.ny, 1)
                    parent_A_4d = (self.A * valid_conflicts).reshape(1, self.nx, self.ny, 1)
                    parent_B_4d = (self.B * valid_conflicts).reshape(1, self.nx, self.ny, 1)
                    parent_E_4d = (self.E * valid_conflicts).reshape(1, self.nx, self.ny, 1)
                    
                    # Calculate offspring resources
                    m_offspring = mx.conv2d(parent_M_4d, kernel_4d, padding=1).squeeze()
                    a_offspring = mx.conv2d(parent_A_4d, kernel_4d, padding=1).squeeze()
                    b_offspring = mx.conv2d(parent_B_4d, kernel_4d, padding=1).squeeze()
                    e_offspring = mx.conv2d(parent_E_4d, kernel_4d, padding=1).squeeze()
                    
                    # Find valid offspring positions
                    offspring_pos = (offspring_max_priority > 0) & has_conflict
                    
                    # Place offspring
                    self.M = mx.where(offspring_pos, m_offspring * 0.5, self.M)
                    self.A = mx.where(offspring_pos, a_offspring * 0.5, self.A)
                    self.B = mx.where(offspring_pos, b_offspring * 0.5, self.B)
                    self.E = mx.where(offspring_pos, e_offspring * 0.5, self.E)
                    self.alive = mx.where(offspring_pos, 1.0, self.alive)
                    self.division_cooldown = mx.where(offspring_pos, self.config.offspring_cooldown, self.division_cooldown)
                    
                    # Update parents
                    self.M = mx.where(winning_parents, self.M * 0.5, self.M)
                    self.A = mx.where(winning_parents, self.A * 0.5, self.A)
                    self.B = mx.where(winning_parents, self.B * 0.5, self.B)
                    self.E = mx.where(winning_parents, self.E * 0.5, self.E)
                    self.division_cooldown = mx.where(winning_parents, self.config.division_cooldown, self.division_cooldown)
                    parent_divided = parent_divided | winning_parents
        
        
    def get_stats(self) -> Dict[str, float]:
        """Get system statistics"""
        n_cells = float(mx.sum(self.alive))
        
        if n_cells == 0:
            return {
                'n_cells': 0,
                'total_M': 0.0,
                'total_A': 0.0,
                'total_B': 0.0,
                'total_E': 0.0,
                'avg_M': 0.0,
                'avg_A': 0.0,
                'avg_B': 0.0,
                'avg_E': 0.0,
                'n_above_threshold': 0,
                'n_on_cooldown': 0,
                'max_M': 0.0,
                'n_ready_to_divide': 0
            }
            
        total_M = float(mx.sum(self.M))
        total_A = float(mx.sum(self.A))
        total_B = float(mx.sum(self.B))
        total_E = float(mx.sum(self.E))
        
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
            'total_A': total_A,
            'total_B': total_B,
            'total_E': total_E,
            'avg_M': total_M / n_cells,
            'avg_A': total_A / n_cells,
            'avg_B': total_B / n_cells,
            'avg_E': total_E / n_cells,
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
        
    def get_cell_values_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get M, A, B, and E values for living cells"""
        alive_np = np.array(self.alive)
        M_np = np.array(self.M)
        A_np = np.array(self.A)
        B_np = np.array(self.B)
        E_np = np.array(self.E)
        
        # Get values only for living cells
        alive_mask = alive_np > 0
        M_values = M_np[alive_mask]
        A_values = A_np[alive_mask]
        B_values = B_np[alive_mask]
        E_values = E_np[alive_mask]
        
        return M_values, A_values, B_values, E_values
        
    @property
    def cell_presence(self):
        """For compatibility with TLBM interface"""
        return self.alive
        
    @property 
    def n_cells(self):
        """For compatibility with sparse interface"""
        return int(mx.sum(self.alive))