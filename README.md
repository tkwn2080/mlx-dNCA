# mlx-dNCA
This is the third form of an ongoing project exploring the role of gradient dissipation in the origins of biology complexity. I am currently rewriting the entire simulation in MLX from the previous variants which were PyTorch-Taichi (for the NCA and TLBM respectively) and then pure Taichi for both parts. The current version uses MLX for both the NCA and TLBM aspects, tailored to my current hardware (M4 Pro Mac Mini) and with the intent of upgrading to an M4 Max Mac Studio later this year.

The core of this project is the coupling of an NCA to a thermal lattice Boltzmann model, hence the title: dissipative neural cellular automata. While there are some cases of dissipative cellular automata (e.g., Rolli & Zambonelli, 2002), this project stemmed directly from Schneider and Kay's (1995) Order from Disorder: The Thermodynamics of Complexity in Biology. The aim, in short, is to use evolved NCAs as a testbed for the hypothesis that biological complexity is driven by exergy dissipation; hence the coupling with a TLBM.

The earliest versions of this project used evolved NCAs to regulate Bénard convection (e.g., Vignon et al., 2023), but lately I have been drawn towards modelling artificial life more directly. The current iteration includes multiple resources, a two-stage autocatalytic cycle, cell division, and gradient-based environment–cell and cell–cell diffusion. This is passive for the moment, with the next step being to fold back in the NCA element and have this activated by the genome: if a given gene is active, then the NCA will act as a controller.

The NCA controller will handle the rate for each stage of the autocatalytic process, as well as allowing for costly active transport of resources (which in the passive form enter the cell only down a gradient). I am also experimenting with allowing for active sharing of resources, energy, and material between cells, and especially with the question of signalling. When it comes to signalling, my eventual intent is to implement a hierarchical NCA model (e.g., Bielawski at al., 2024) with signalling seeming a natural role for this element.

All of these cases will be evolved with exergy dissipation as the sole measure of fitness, although I am also considering empowerment in a multi-objective scenario. Currently I am using CMA-ES due to my familiarity with this algorithm, but once the MLX-based system is stable I will explore alternatives (e.g., AFPO as used by Bielawski et al., 2024), including multi-objective optimisation. There also remains some uncertainty about how exactly exergy is to be characterised in such a system, and I am continuing theoretical research in this direction.

The central difficulty in this project is the massive expressibility of NCAs, which elsewhere has been an advantage but here requires one to design a system of physical law to constrain this process. The central role for NCAs in this case is to serve as an abstraction over (for now, feedforward) chemical processing pathways, expressing any possible relation between inputs (receptors) and outputs (effectors). This may need to be constrained, however, by something like a conservation of network activations, whether directly enforced or evolved.

Much of the work here, and this increasingly with each iteration, has been tuning the basic variables in order to create a situation where the evolution of biological complexity is possible but not necessarily preordained. This has been the most obscure and complex element of the project so far, with the core of this being fed by the continued theoretical research which constitutes the core of my doctoral work.

## References

- Bielawski, K., Gaylinn, N., Lunn, C., Motia, K., & Bongard, J. (2024). Evolving Hierarchical Neural Cellular Automata. In *Proceedings of the Genetic and Evolutionary Computation Conference* (pp. 78-86).

- Roli, A., & Zambonelli, F. (2002). Emergence of Macro Spatial Structures in Dissipative Cellular Automata. In S. Bandini, B. Chopard, & M. Tomassini (Eds.), *Cellular Automata. ACRI 2002. Lecture Notes in Computer Science* (Vol. 2493, pp. 144-155). Springer, Berlin, Heidelberg.

- Schneider, E. D., & Kay, J. J. (1995). Order from Disorder: The Thermodynamics of Complexity in Biology. In M. P. Murphy & L. A. J. O'Neill (Eds.), *What is Life: The Next Fifty Years. Reflections on the Future of Biology* (pp. 161-172). Cambridge University Press.

- Vignon, C., Rabault, J., Vasanth, J., Alcántara-Ávila, F., Mortensen, M., & Vinuesa, R. (2023). Effective control of two-dimensional Rayleigh–Bénard convection: Invariant multi-agent reinforcement learning is all you need. *Physics of Fluids*, 35(6).
