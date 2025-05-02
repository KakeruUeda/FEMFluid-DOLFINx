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

class Domain:
    """ 
    Domain class.
    Only for gmsh. 
    """
    def __init__(self, cfg):
        msh_path = Path(
            to_absolute_path(cfg.mesh.filepath)
        )
        self.msh_info = gmshio.read_from_msh(
            str(msh_path), MPI.COMM_WORLD
        )

        self.msh = self.msh_info.mesh
        self.cell_tags = self.msh_info.cell_tags
        self.facet_tags = self.msh_info.facet_tags

        self.deg_u = cfg.mesh.deg_u
        self.deg_p = cfg.mesh.deg_p

        # define integration measures
        self.dx = ufl.Measure(
            'dx', 
            domain = self.msh, 
            metadata = {'quadrature_degree': 4}
        )
        self.ds = ufl.Measure(
            'ds', 
            domain = self.msh, 
            subdomain_data = self.facet_tags, 
            metadata={'quadrature_degree': 4}
        )

def define_functionspace(domain):
    V = fem.functionspace(
        domain.msh, element(
            "Lagrange", 
            domain.msh.basix_cell(), 
            domain.deg_u,
            shape=(domain.msh.geometry.dim,)
        )
    )
    Q = fem.functionspace(
        domain.msh, element(
            "Lagrange",
            domain.msh.basix_cell(),
            domain.deg_p
        )
    )
    return [V, Q]