"""
Centralized configuration for the coupled TLBM-Cell system
This is the single source of truth for all parameters
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TLBMConfig:
    """Configuration for the Thermal Lattice Boltzmann Method"""
    # Rayleigh and Prandtl numbers (determine flow regime)
    Ra: float = 5e4           # Rayleigh number
    Pr: float = 0.71          # Prandtl number (air-like)

    # Temperature boundary conditions
    T_hot: float = 1.0        # Bottom boundary temperature
    T_cold: float = 0.0       # Top boundary temperature

    # Resource parameters (now for A and B)
    resource_kappa: float = 0.02      # Resource diffusivity (same for A and B)
    resource_A_boundary_value: float = 1.0   # Resource A concentration at source
    resource_B_boundary_value: float = 1.0   # Resource B concentration at source
    resource_boundary_top: float = 0.0      # Resource concentration at top (sink)
    
    # Resource source allocation
    resource_source_seed: int = 42    # Random seed for A/B left/right allocation
    resource_source_left: Optional[str] = None   # Will be 'A' or 'B' after init
    resource_source_right: Optional[str] = None  # Will be 'B' or 'A' after init

    # Cell interaction parameters
    cell_resistance: float = 1.0      # Flow resistance from cells (0=no effect, 1=complete blocking)

    # Numerical parameters
    tau_f: Optional[float] = None     # Relaxation time for flow (computed from Ra, Pr)
    tau_g: Optional[float] = None     # Relaxation time for temperature (computed from Ra, Pr)
    tau_r: Optional[float] = None     # Relaxation time for resources
    gravity: Optional[float] = None   # Gravity strength (hardcoded for stability)
    niu: Optional[float] = None       # Kinematic viscosity (computed)
    kappa: Optional[float] = None     # Thermal diffusivity (computed)
    beta: float = 0.5                 # Thermal expansion coefficient

    def __post_init__(self):
        """Compute derived parameters from Ra and Pr"""
        # For a domain of height H=1 in lattice units
        # Ra = g*beta*DT*H^3/(niu*kappa)
        # Pr = niu/kappa

        # Target Mach number for stability
        u_max_target = 0.1
        DT = self.T_hot - self.T_cold

        # Compute kinematic viscosity and thermal diffusivity
        # From Ra and Pr definitions
        niu = ((self.Ra / DT) ** (-1/3)) * (self.Pr ** (2/3))
        kappa = niu / self.Pr

        # Relaxation times (tau = 3*diffusivity + 0.5)
        self.tau_f = 3.0 * niu + 0.5
        self.tau_g = 3.0 * kappa + 0.5
        self.tau_r = 3.0 * self.resource_kappa + 0.5

        # Gravity - hardcoded for stability instead of computing from Ra
        # Computing from Ra gives unrealistically high values that cause instability
        # Common LBM implementations use gravity values around 0.001-0.01
        self.gravity = 0.03  # Increased for clear Bénard cell formation (Ra ≈ 2100)

        # Show effective Ra for reference
        Ra_effective = self.gravity * 1.0 * DT * 1.0**3 / (niu * kappa)  # beta=1, H=1

        # Check stability
        if self.tau_f <= 0.5 or self.tau_g <= 0.5:
            print(f"Warning: Relaxation times too small for stability!")
            print(f"tau_f = {self.tau_f:.3f}, tau_g = {self.tau_g:.3f}")
            
        # Randomly allocate A/B to left/right sources
        import mlx.core as mx
        mx.random.seed(self.resource_source_seed)
        if mx.random.uniform() > 0.5:
            self.resource_source_left = 'A'
            self.resource_source_right = 'B'
        else:
            self.resource_source_left = 'B'
            self.resource_source_right = 'A'


@dataclass
class CellConfig:
    """Configuration for cell behavior"""
    # Reaction parameters (per step, not per unit time)
    reaction_rate_1: float = 0.05        # A + E -> 2E reaction rate per step
    reaction_rate_2: float = 0.05        # B + E -> M reaction rate per step
    
    # Resource parameters
    diffusion_rate: float = 0.1          # Internal diffusion of A,B between cells per step
    ingestion_rate: float = 1.0          # Rate of resource uptake per step (same for A and B)
    
    # Energy parameters
    initial_E: float = 0.5               # Initial energy level for new cells
    death_threshold_E: float = 0.01      # E below this causes cell death
    
    # Saturation limits
    max_A: float = 2.0                   # Maximum A concentration per cell
    max_B: float = 2.0                   # Maximum B concentration per cell  
    max_E: float = 3.0                   # Maximum E concentration per cell

    # Maintenance and degradation
    maintenance_rate: float = 0.01       # M consumed per step for maintenance

    # Cell mechanics
    division_threshold: float = 2.0       # M threshold for division
    min_m_after_division: float = 0.5     # Minimum M retained after division
    death_threshold_M: float = 0.01       # M below this causes cell death
    division_cooldown: float = 50         # Steps before parent cell can divide again
    offspring_cooldown: float = 10        # Steps before offspring cell can divide

    # Stochastic parameters
    stochastic_update_prob: float = 0.5   # Probability of stochastic updates
    division_probability: float = 0.5     # Probability of dividing when ready


@dataclass
class CouplingConfig:
    """Configuration for cell-TLBM coupling strengths"""
    # Resource coupling
    resource_extraction_factor: float = 0.01   # How much extracted resources affect TLBM

    # Update frequencies
    cell_update_interval: float = 10          # Cell steps every N TLBM steps


@dataclass
class SimulationConfig:
    """Overall simulation configuration"""
    # Grid dimensions
    nx: int = 128
    ny: int = 64

    # Sub-configurations
    tlbm: TLBMConfig = None
    cells: CellConfig = None
    coupling: CouplingConfig = None

    # Visualization
    update_interval: int = 20      # Animation update interval

    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.tlbm is None:
            self.tlbm = TLBMConfig()
        if self.cells is None:
            self.cells = CellConfig()
        if self.coupling is None:
            self.coupling = CouplingConfig()


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig(
    nx=128,
    ny=64,
    tlbm=TLBMConfig(
        Ra=5e4,
        Pr=0.71,
        T_hot=1.0,
        T_cold=0.0,
        resource_kappa=0.02,
        resource_A_boundary_value=1.0,
        resource_B_boundary_value=1.0,
        resource_source_seed=42,
        cell_resistance=1.0
    ),
    cells=CellConfig(
        reaction_rate_1=0.05,      # A + E -> 2E
        reaction_rate_2=0.05,      # B + E -> M
        diffusion_rate=0.04,
        ingestion_rate=0.1,
        initial_E=0.5,
        death_threshold_E=0.01,
        max_A=2.0,
        max_B=2.0,
        max_E=3.0,
        maintenance_rate=0.001,
        division_threshold=2.0,
        min_m_after_division=0.5,
        death_threshold_M=0.01,
        division_cooldown=50,
        offspring_cooldown=10,
        stochastic_update_prob=0.5,
        division_probability=0.3
    ),
    coupling=CouplingConfig(
        resource_extraction_factor=0.01,
        cell_update_interval=10
    )
)
