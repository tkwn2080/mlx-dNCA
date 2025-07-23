"""
Example of optimized boundary condition application using MLX compilation
This shows how to batch operations and use mx.compile for better performance
"""

import mlx.core as mx

@mx.compile
def apply_thermal_boundaries(g, T_hot, T_cold, w):
    """
    Apply temperature boundary conditions in a single compiled function.
    More efficient than setting individual slices.
    """
    # Create boundary value arrays
    w_broadcast = w.reshape(1, 1, 9)
    
    # Create masks for boundaries
    nx, ny, _ = g.shape
    
    # Bottom boundary mask (y=0 and y=1)
    bottom_mask = mx.zeros((nx, ny), dtype=mx.bool_)
    bottom_mask[:, 0:2] = True
    
    # Top boundary mask (y=ny-2 and y=ny-1)
    top_mask = mx.zeros((nx, ny), dtype=mx.bool_)
    top_mask[:, -2:] = True
    
    # Apply boundaries using where
    g_new = mx.where(bottom_mask[:, :, None], w_broadcast * T_hot, g)
    g_new = mx.where(top_mask[:, :, None], w_broadcast * T_cold, g_new)
    
    return g_new


@mx.compile
def apply_resource_boundaries_optimized(r_A, r_B, w, nx, ny, mid_x, 
                                      resource_A_val, resource_B_val,
                                      source_on_left_is_A):
    """
    Apply resource boundary conditions with better vectorization.
    """
    w_broadcast = w.reshape(1, 1, 9)
    
    # Create coordinate grids
    x_grid = mx.arange(nx)[:, None]
    y_grid = mx.arange(ny)[None, :]
    
    # Define source regions
    left_source = (x_grid >= 1) & (x_grid < mid_x) & (y_grid >= 0) & (y_grid < 2)
    right_source = (x_grid >= mid_x) & (x_grid < nx-1) & (y_grid >= 0) & (y_grid < 2)
    
    # Define sink region (top)
    sink_region = (y_grid >= ny-2)
    
    if source_on_left_is_A:
        # Left: A source, Right: B source
        r_A = mx.where(left_source[:, :, None], w_broadcast * resource_A_val, r_A)
        r_B = mx.where(right_source[:, :, None], w_broadcast * resource_B_val, r_B)
        # Clear opposite resources
        r_B = mx.where(left_source[:, :, None], w_broadcast * 0.0, r_B)
        r_A = mx.where(right_source[:, :, None], w_broadcast * 0.0, r_A)
    else:
        # Left: B source, Right: A source
        r_B = mx.where(left_source[:, :, None], w_broadcast * resource_B_val, r_B)
        r_A = mx.where(right_source[:, :, None], w_broadcast * resource_A_val, r_A)
        # Clear opposite resources
        r_A = mx.where(left_source[:, :, None], w_broadcast * 0.0, r_A)
        r_B = mx.where(right_source[:, :, None], w_broadcast * 0.0, r_B)
    
    # Apply sink conditions
    r_A = mx.where(sink_region[:, :, None], w_broadcast * 0.0, r_A)
    r_B = mx.where(sink_region[:, :, None], w_broadcast * 0.0, r_B)
    
    return r_A, r_B


@mx.compile
def apply_periodic_ghost_nodes(f, g, r_A, r_B):
    """
    Apply x-periodic boundary conditions using ghost nodes.
    Compiled for better performance.
    """
    # Left ghost (x=0) = right physical (x=nx-2)
    f = mx.concatenate([f[-2:, :, :], f[1:, :, :]], axis=0)
    g = mx.concatenate([g[-2:, :, :], g[1:, :, :]], axis=0)
    r_A = mx.concatenate([r_A[-2:, :, :], r_A[1:, :, :]], axis=0)
    r_B = mx.concatenate([r_B[-2:, :, :], r_B[1:, :, :]], axis=0)
    
    # Right ghost (x=nx-1) = left physical (x=1)
    f = mx.concatenate([f[:-1, :, :], f[1:2, :, :]], axis=0)
    g = mx.concatenate([g[:-1, :, :], g[1:2, :, :]], axis=0)
    r_A = mx.concatenate([r_A[:-1, :, :], r_A[1:2, :, :]], axis=0)
    r_B = mx.concatenate([r_B[:-1, :, :], r_B[1:2, :, :]], axis=0)
    
    return f, g, r_A, r_B


@mx.compile
def apply_all_boundaries(f, g, r_A, r_B, T_hot, T_cold, w, nx, ny, mid_x,
                        resource_A_val, resource_B_val, source_on_left_is_A):
    """
    Apply all boundary conditions in a single compiled function.
    This reduces function call overhead and improves GPU utilization.
    """
    # Temperature boundaries
    g = apply_thermal_boundaries(g, T_hot, T_cold, w)
    
    # Resource boundaries
    r_A, r_B = apply_resource_boundaries_optimized(
        r_A, r_B, w, nx, ny, mid_x, 
        resource_A_val, resource_B_val, source_on_left_is_A
    )
    
    # Periodic boundaries
    f, g, r_A, r_B = apply_periodic_ghost_nodes(f, g, r_A, r_B)
    
    # Bounce-back for fluid (simplified version)
    # Bottom boundary
    f[:, 0, 2] = f[:, 1, 4]   # North at ghost = South at wall
    f[:, 0, 5] = f[:, 1, 7]   # NE at ghost = SW at wall
    f[:, 0, 6] = f[:, 1, 8]   # NW at ghost = SE at wall
    
    # Top boundary
    f[:, -1, 4] = f[:, -2, 2]  # South at ghost = North at wall
    f[:, -1, 7] = f[:, -2, 5]  # SW at ghost = NE at wall
    f[:, -1, 8] = f[:, -2, 6]  # SE at ghost = NW at wall
    
    return f, g, r_A, r_B


# Example of how to use in TLBM class
class OptimizedTLBM:
    def __init__(self, nx, ny, config):
        # ... initialization ...
        
        # Pre-compile boundary function with partial application
        self.apply_boundaries = lambda f, g, r_A, r_B: apply_all_boundaries(
            f, g, r_A, r_B, 
            config.T_hot, config.T_cold, self.w,
            self.nx, self.ny, self.nx_phys // 2 + 1,
            config.resource_A_boundary_value,
            config.resource_B_boundary_value,
            config.resource_source_left == 'A'
        )