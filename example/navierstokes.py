# -*- coding: utf-8 -*-

"""
Incompressible Navier-Stokes Solver.

This module solve the incompressible Navier-Stokes
equations using finite element methods.

Author: Kakeru Ueda
Date: 2025-04-27
License: MIT License
"""

# File: navierstokes.py

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
    assemble_scalar 
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile

import ufl

import basix

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Takes in hydra params.
    e.g. python navier_stokes.py time.dt = 0.01 fluid.Re = 100 ...
    """
    
    print("----- Configuration -----")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
if __name__ == "__main__":
    main()
