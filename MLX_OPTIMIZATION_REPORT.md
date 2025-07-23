# MLX Optimization Report for mlx-dNCA

## Executive Summary
This report identifies opportunities to better utilize MLX features in the mlx-dNCA codebase. The main areas for improvement include reducing NumPy conversions, adding more JIT compilation, fixing MLX-NumPy conversions that break the computation graph, and ensuring proper array evaluation.

## 1. Visualization Operations in MLX

### Current Issue
In `main.py` and `src/tlbm.py`, gradient and clipping operations are done with NumPy after conversion:

```python
# main.py lines 55, 181
vort = np.gradient(fields['vy'], axis=0) - np.gradient(fields['vx'], axis=1)

# main.py lines 66-67, 186-187  
rgb_resources[:, :, 0] = np.clip(fields['P_A'].T, 0, 1)
rgb_resources[:, :, 2] = np.clip(fields['P_B'].T, 0, 1)
```

### Recommendation
Compute these in MLX before conversion:

```python
# In tlbm.py get_numpy_fields() method, add:
def get_numpy_fields(self):
    # Compute derived fields in MLX
    vx_phys = self.vel[self.phys_x, self.phys_y, 0]
    vy_phys = self.vel[self.phys_x, self.phys_y, 1]
    
    # Compute vorticity in MLX
    dvx_dy = mx.gradient(vx_phys, axis=1)
    dvy_dx = mx.gradient(vy_phys, axis=0)
    vorticity = dvy_dx - dvx_dy
    
    # Clip resources in MLX
    P_A_clipped = mx.clip(self.P_A[self.phys_x, self.phys_y], 0, 1)
    P_B_clipped = mx.clip(self.P_B[self.phys_x, self.phys_y], 0, 1)
    
    # Ensure evaluation before conversion
    mx.eval(vorticity, P_A_clipped, P_B_clipped)
    
    return {
        'T': np.array(self.T[self.phys_x, self.phys_y]),
        'vx': np.array(vx_phys),
        'vy': np.array(vy_phys),
        'vorticity': np.array(vorticity),
        'P_A': np.array(self.P_A[self.phys_x, self.phys_y]),
        'P_B': np.array(self.P_B[self.phys_x, self.phys_y]),
        'P_A_clipped': np.array(P_A_clipped),
        'P_B_clipped': np.array(P_B_clipped),
        'rho': np.array(self.rho[self.phys_x, self.phys_y])
    }
```

## 2. Missing JIT Compilation Opportunities

### Current Issue
Only 2 functions use `@mx.compile` in the entire codebase.

### Recommendation
Add compilation to performance-critical functions:

#### In `src/cell.py`:
```python
@mx.compile
def compute_reactions(M, A, B, E, alive, config):
    """Compile the reaction computation"""
    reaction_1_amount = config.reaction_rate_1 * A * E * alive
    reaction_2_amount = config.reaction_rate_2 * B * E * alive
    return reaction_1_amount, reaction_2_amount

@mx.compile
def apply_diffusion(A, B, alive, kernel, diffusion_rate):
    """Compile resource diffusion"""
    A_alive = A * alive
    B_alive = B * alive
    
    kernel_4d = kernel.reshape(1, 3, 3, 1)
    
    A_4d = A_alive.reshape(1, A.shape[0], A.shape[1], 1)
    A_diffusion = mx.conv2d(A_4d, kernel_4d, padding=1).squeeze()
    
    B_4d = B_alive.reshape(1, B.shape[0], B.shape[1], 1)
    B_diffusion = mx.conv2d(B_4d, kernel_4d, padding=1).squeeze()
    
    return A_diffusion, B_diffusion
```

#### In `src/tlbm.py`:
```python
@mx.compile
def apply_boundary_conditions(f, g, r_A, r_B, T_hot, T_cold, w, 
                            resource_A_val, resource_B_val, mid_x):
    """Compile boundary condition application"""
    # Implementation of _apply_ghost_boundaries
    ...
```

## 3. NumPy Conversion in Cell Division

### Current Issue
In `src/cell.py` lines 280-293, the code converts MLX arrays to NumPy for conflict resolution:

```python
# Convert to numpy for conflict resolution
conflict_positions = np.argwhere(np.array(has_conflict, copy=False))
```

### Recommendation
Stay in MLX using `mx.where` and vectorized operations:

```python
# Replace the entire conflict resolution section with MLX operations
if mx.any(has_conflict):
    # Get conflict positions using MLX
    conflict_indices = mx.where(has_conflict)
    n_conflicts = len(conflict_indices[0])
    
    if n_conflicts > 0:
        # Process conflicts in MLX
        for i in range(n_conflicts):
            cx = conflict_indices[0][i]
            cy = conflict_indices[1][i]
            
            # Use MLX operations to handle the conflict
            # ... (vectorized MLX implementation)
```

## 4. Comprehensive Array Evaluation

### Current Issue
Only specific arrays are evaluated, which could lead to deferred computation buildup.

### Recommendation
Add periodic comprehensive evaluation:

```python
# In src/cell.py step() method, add at the end:
if self.step_counter % 100 == 0:  # Every 100 steps
    # Evaluate all major arrays
    mx.eval(self.M, self.A, self.B, self.E, self.alive,
            self.resource_A_extraction, self.resource_B_extraction,
            self.division_cooldown)

# In src/tlbm.py step() method:
if self.step_counter % 100 == 0:
    # Evaluate all distribution functions and fields
    mx.eval(self.f, self.g, self.r_A, self.r_B,
            self.rho, self.vel, self.T, self.P_A, self.P_B)
```

## 5. Batched Operations for Better GPU Utilization

### Current Issue
Some operations are done sequentially when they could be batched.

### Recommendation
Batch similar operations:

```python
# In src/tlbm.py _apply_ghost_boundaries():
# Instead of separate operations for each resource, batch them:
@mx.compile
def set_resource_boundaries(r_A, r_B, w, mid_x, resource_A_val, resource_B_val):
    """Batch resource boundary updates"""
    w_broadcast = w.reshape(1, 1, 9)
    
    # Create masks for left/right regions
    left_mask = mx.zeros_like(r_A[:, :, 0])
    right_mask = mx.zeros_like(r_A[:, :, 0])
    left_mask[1:mid_x, 0:2] = 1.0
    right_mask[mid_x:, 0:2] = 1.0
    
    # Apply boundaries using masks (vectorized)
    r_A_new = mx.where(left_mask[:, :, None], w_broadcast * resource_A_val, r_A)
    r_B_new = mx.where(right_mask[:, :, None], w_broadcast * resource_B_val, r_B)
    
    return r_A_new, r_B_new
```

## 6. Memory Layout Optimization

### Current Issue
Some reshape operations could be avoided by using consistent memory layouts.

### Recommendation
Consider using consistent 4D tensors throughout for convolution operations:

```python
# Keep distributions in 4D format (batch, height, width, channels)
# This avoids repeated reshaping for convolutions
```

## Implementation Priority

1. **High Priority**: Fix NumPy conversions in cell division (breaks computation graph)
2. **High Priority**: Add mx.compile to frequently called functions
3. **Medium Priority**: Move visualization preprocessing to MLX
4. **Medium Priority**: Add comprehensive evaluation points
5. **Low Priority**: Optimize memory layouts and batching

## Expected Performance Improvements

- **JIT Compilation**: 10-30% speedup for compiled functions
- **Reduced NumPy conversions**: 5-10% overall speedup
- **Better GPU utilization**: 15-25% speedup for batched operations
- **Memory efficiency**: Reduced memory transfers between CPU/GPU

## Testing Recommendations

After implementing optimizations:
1. Profile with MLX's built-in profiler
2. Compare step rates before/after
3. Monitor memory usage
4. Verify numerical accuracy is maintained