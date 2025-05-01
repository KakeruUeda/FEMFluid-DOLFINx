# -*- coding: utf-8 -*-

"""
Incompressible Navier-Stokes Solver.
===========================================================

Solves the unsteady incompressible Navier-Stokes equations 
using **dolfinx**.  
(see: https://docs.fenicsproject.org/dolfinx/main/python/)
The script is structured so that every numerical / physical 
parameter can be supplied from the *Hydra* configuration
system.

Features
-------------
* Time integration: Fractional step (projection) method
* I/O: XDMF time series for velocity & pressure

Author: Kakeru Ueda
Date: 2025-04-27
License: MIT
"""

import dolfinx
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import (
    fem, 
    mesh, 
    io, 
    plot, 
    log
)
from dolfinx.fem import (
    Constant, 
    dirichletbc, 
    Function, 
    functionspace, 
    Expression, 
    form, 
    assemble_scalar,
    petsc
)
from dolfinx.fem.petsc import (
    assemble_matrix, 
    create_vector, 
    assemble_vector, 
    apply_lifting, 
    set_bc
)
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, gmshio

import ufl

import basix
from basix.ufl import element
from hydra.utils import to_absolute_path
from pathlib import Path
from dataclasses import dataclass

from hydra.utils import to_absolute_path
from pathlib import Path

# femfluid/solver.py
import logging

import src.domain as domain_
import src.solver as solver_
import src.io as io_

log = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Takes in hydra params.
    e.g. python navier_stokes.py time.dt = 0.01 fluid.Re = 100 ...
    """
    
    print("----- Configuration -----")
    print(OmegaConf.to_yaml(cfg, resolve = True))

    d = domain_.Domain(cfg)

    [V, Q] = domain_.define_functionspace(d)

    bcu = solver_.make_velocity_bcu(V, d.facet_tags, cfg)
    # bcp = solver_.make_pressure_anchor(Q, d.msh, cfg)
    bcp = solver_.make_pressure_bcp(Q, d.facet_tags, cfg)

    ns = solver_.NavierStokes(
        V, Q,
        Re = cfg.fluid.Re,
        t_end = cfg.time.t_end,
        dt = cfg.time.dt,
        dx = d.dx,
        bcu = bcu,
        bcp = bcp
    )

    ns.build_weakform(d.dx)
    ns.setup_solver_petsc(d.msh)

    [ufile, pfile] = io_.open_files_xdfm()
    [u_vis, p_vis] = io_.get_vars_visu(d)

    ufile.write_mesh(d.msh)
    pfile.write_mesh(d.msh)

    io_.output_velocity(ns.u_, u_vis, ufile, 0)
    io_.output_pressure(ns.p_, p_vis, pfile, 0)

    time_now = 0.0

    num_steps = np.int32(ns.t_end / ns.dt) + 1

    for step in range(1, num_steps + 1):
        time_now += ns.dt

        print("\n\n Load step = ", step)
        print("     Time      = ", time_now)

        # solve system of eqs. (projection).
        ns.solve_intermidiate()
        ns.solve_poisson()
        ns.solve_correction()

        ns.update_variables()
        
        if step % cfg.solver.output_interval == 0:
            io_.output_velocity(ns.u_, u_vis, ufile, time_now)
            io_.output_pressure(ns.p_, p_vis, pfile, time_now)
    
    ufile.close()
    pfile.close()

    print("----- Terminated. -----")

if __name__ == "__main__":
    main()
