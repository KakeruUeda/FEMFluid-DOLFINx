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

class Domain:
    """ 
    Domain Interface.
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

class NavierStokes(Domain):
    """ N.S. eq. handler."""

    def __init__(self, cfg):
        """ Setup variables. """

        super().__init__(cfg)

        self.Re = cfg.fluid.Re
        self.t_end = cfg.time.t_end
        self.dt = cfg.time.dt
        self.deg_u = cfg.mesh.deg_u
        self.deg_p = cfg.mesh.deg_p

        self.num_steps = np.int32(
            self.t_end / self.dt
        ) + 1

    def define_space(self):
        elem_v = element(
            "Lagrange", 
            self.msh.basix_cell(), 
            self.deg_u, 
            shape=(self.msh.geometry.dim,)
        )
        elem_p = element(
            "Lagrange", 
            self.msh.basix_cell(), 
            self.deg_p
        )

        self.V = functionspace(self.msh, elem_v)
        self.Q = functionspace(self.msh, elem_p)

        # variables needed in forms
        (self.u, self.p) = ufl.TrialFunction(self.V), ufl.TrialFunction(self.Q)
        (self.v, self.q) = ufl.TestFunction(self.V), ufl.TestFunction(self.Q)

        (self.u_, self.p_) = Function(self.V), Function(self.Q)
        (self.u_n, self.p_n) = Function(self.V), Function(self.Q)

    def set_dirichlet_bcs(self, cfg):
        """ set dirichlet boundary conditions"""

        def uniform_expression(dir, value):
            print(dir)
            print(value)
            bc = fem.dirichletbc(
                value = value, 
                dofs = dofs, 
                V = self.V.sub(dir)
            )
            return bc
        
        class Poiseuille:
            def __init__(self, cfg):
                self.max_vel = cfg.value
                self.center = cfg.center
                self.radius = cfg.radius
                self.normal = cfg.normal

            def ux(self, x):
                value = 1 - (2 * x[0] / self.radius - 1)**2
                return [value]

            def uy(self, x):
                value = 1 - (2 * x[0] / self.radius - 1)**2
                return [value]

        def waveform_expression(value, poiseuille): 
            wx, wy = Function(self.V), Function(self.V)
            wx.interpolate(poiseuille.ux)
            wy.interpolate(poiseuille.uy)

            bcx = fem.dirichletbc(
                wx, 
                dofs = dofs, 
                V = self.V.sub(0)
            )
            bcy = fem.dirichletbc(
                wy, 
                dofs = dofs, 
                V = self.V.sub(1)
            )

            return [bcx, bcy]

        self.bcu = []
        for bc in cfg.boundary.dirichlet.velocity:
            tag, type, dir, value = bc

            dofs = fem.locate_dofs_topological(
                self.V.sub(dir), 
                self.facet_tags.dim, 
                self.facet_tags.find(tag)
            )

            if type == "uniform":
                bc = uniform_expression(dir, value)
                self.bcu.append(bc)
            elif type == "poiseuille":
                poiseuille = Poiseuille(cfg)
                bc = waveform_expression(value, poiseuille)
                self.bcu.extend(bc)

        tag, value = cfg.boundary.dirichlet.pressure[0]

        
        def near_anchor_point(x):
            return np.isclose(x[0], 0) & np.isclose(x[1], 0)
        
        anchor_vertices = mesh.locate_entities(
            self.msh, 
            dim = 0, 
            marker = near_anchor_point
        )

        dofs_anchor = fem.locate_dofs_topological(
            self.Q, 
            0, 
            anchor_vertices
        )
            
        bc_anchor = fem.dirichletbc(
            value = value, 
            dofs = dofs_anchor, 
            V = self.Q
        )

        self.bcp = [bc_anchor]

    def set_projection_matrix(self):
        """ projection matrix """
        # Step 1
        self.lhs1 = form(
            ufl.dot(self.u / self.dt, self.v) * self.dx
            + 0.5 * (1.0 / self.Re) 
            * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) 
            * self.dx
        )

        self.rhs1 = form(
            ufl.dot(
                self.u_n / self.dt, 
                self.v
            ) * self.dx
            - ufl.dot(
                ufl.dot(self.u_n, ufl.nabla_grad(self.u_n)), 
                self.v
            ) * self.dx
            - 0.5 * (1.0 / self.Re) * 
            ufl.inner(ufl.grad(self.u_n), ufl.grad(self.v)) 
            * self.dx
        )

        self.A1 = petsc.assemble_matrix(self.lhs1, bcs = self.bcu)
        self.A1.assemble()
        self.b1 = petsc.create_vector(self.rhs1)
    
        # Step 1
        self.lhs2 = form(
            ufl.inner(ufl.grad(self.p), ufl.grad(self.q)) * self.dx 
        )
        self.rhs2 = form(
            -(1.0 / self.dt) * ufl.div(self.u_) * self.q * self.dx
        )
   
        self.A2 = petsc.assemble_matrix(self.lhs2, bcs = self.bcp)
        self.A2.assemble()
        self.b2 = petsc.create_vector(self.rhs2)

        # Step 2
        self.lhs3 = form(
            ufl.dot(self.u, self.v) * self.dx
        )
        self.rhs3 = form(
            ufl.dot(self.u_, self.v) * self.dx 
            - self.dt * ufl.dot(ufl.nabla_grad(self.p_), self.v) * self.dx
        )

        # Step 3
        self.A3 = assemble_matrix(self.lhs3)
        self.A3.assemble()
        self.b3 = create_vector(self.rhs3)

    def set_solver_petsc(self):
        """ petsc setting """
        # Solver for step 1
        self.solver1 = PETSc.KSP().create(self.msh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        self.pc1 = self.solver1.getPC()
        self.pc1.setType(PETSc.PC.Type.HYPRE)
        self.pc1.setHYPREType("boomeramg")

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(self.msh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        self.pc2 = self.solver2.getPC()
        self.pc2.setType(PETSc.PC.Type.HYPRE)
        self.pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(self.msh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        self.pc3 = self.solver3.getPC()
        self.pc3.setType(PETSc.PC.Type.SOR)

    def solve_intermidiate(self):
        # Step 1: Temporal veolcity step
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.rhs1)
        apply_lifting(self.b1, [self.lhs1], [self.bcu])
        self.b1.ghostUpdate(
            addv = PETSc.InsertMode.ADD_VALUES, 
            mode=PETSc.ScatterMode.REVERSE
        )
        set_bc(self.b1, self.bcu)
        self.solver1.solve(self.b1, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

    def solve_poisson(self):
        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.rhs2)
        apply_lifting(self.b2, [self.lhs2], [self.bcp])
        self.b2.ghostUpdate(
            addv = PETSc.InsertMode.ADD_VALUES, 
            mode = PETSc.ScatterMode.REVERSE
        )
        set_bc(self.b2, self.bcp)
        self.solver2.solve(self.b2, self.p_.x.petsc_vec)
        self.p_.x.scatter_forward()

    def solve_correction(self):
        # Step 3: Velocity correction step
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.rhs3)
        self.b3.ghostUpdate(
            addv = PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE
        )
        self.solver3.solve(self.b3, self.u_.x.petsc_vec)
        self.u_.x.scatter_forward()

    def update_variables(self):
        # Update variable with solution form this time step
        self.u_n.x.array[:] = self.u_.x.array[:]
        self.p_n.x.array[:] = self.p_.x.array[:]


def open_files_xdfm():
    ufile = XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w")
    pfile = XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w")

    return [ufile, pfile]

def get_vars_visu(pde):
    V_vis = element(
        "Lagrange",
        pde.msh.basix_cell(),
        pde.deg_u,
        shape = (pde.msh.geometry.dim,),
    )
    Q_vis = element(
        "Lagrange",
        pde.msh.basix_cell(),
        pde.deg_u,
    )
    u_vis = Function(functionspace(pde.msh, V_vis))
    p_vis = Function(functionspace(pde.msh, Q_vis))

    return [u_vis, p_vis]

def output_velocity(u_, u_vis, ufile, t):
    u_vis.interpolate(u_)
    ufile.write_function(u_vis, t)

def output_pressure(p_, p_vis, pfile, t):
    p_vis.interpolate(p_)
    pfile.write_function(p_vis, t)

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Takes in hydra params.
    e.g. python navier_stokes.py time.dt = 0.01 fluid.Re = 100 ...
    """
    
    print("----- Configuration -----")
    print(OmegaConf.to_yaml(cfg, resolve = True))
    
    # problem setting
    ns = NavierStokes(cfg)
    ns.define_space()
    ns.set_dirichlet_bcs(cfg)
    ns.set_projection_matrix()
    ns.set_solver_petsc()

    [ufile, pfile] = open_files_xdfm()
    [u_vis, p_vis] = get_vars_visu(ns)

    ufile.write_mesh(ns.msh)
    pfile.write_mesh(ns.msh)

    output_velocity(ns.u_, u_vis, ufile, 0)
    output_pressure(ns.p_, p_vis, pfile, 0)

    time_now = 0.0

    for step in range(1, ns.num_steps + 1):
        time_now += ns.dt

        print("\n\n Load step = ", step)
        print("     Time      = ", time_now)

        # projection solvers.
        ns.solve_intermidiate()
        ns.solve_poisson()
        ns.solve_correction()

        ns.update_variables()
        
        if step % cfg.solver.output_interval == 0:
            output_velocity(ns.u_, u_vis, ufile, time_now)
            output_pressure(ns.p_, p_vis, pfile, time_now)
    
    ufile.close()
    pfile.close()

    print("----- Terminated. -----")

if __name__ == "__main__":
    main()
