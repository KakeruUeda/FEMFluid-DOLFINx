# femfluid/projection.py
import dolfinx
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import (
    fem, 
    mesh
)
from dolfinx.fem import (
    form, 
    petsc
)
from dolfinx.fem.petsc import (
    assemble_matrix, 
    create_vector, 
    assemble_vector, 
    apply_lifting, 
    set_bc
)
from dolfinx.mesh import MeshTags

import ufl


class PoiseuilleProfile():
    """ poiseuille profile"""

    def __init__(
        self, 
        max_vel: float, 
        radius: float, 
        center: float,
        normal: float
    ):
        self.max_vel = max_vel 
        self.radius = radius 
        self.center = center
        self.normal = normal

    def calc_distance(self, x):
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        return np.sqrt(dx**2 + dy**2)

    def ux(self, x):
        dist = self.calc_distance(x)
        value = 1 - (2 * dist / self.radius - 1)**2
        return [value * self.normal[0]]

    def uy(self, x):
        dist = self.calc_distane(x)
        value = 1 - (2 * dist / self.radius - 1)**2
        return [value * self.normal[1]]
    

def make_velocity_bcu(
    V: fem.FunctionSpace,
    facet_tags: MeshTags,
    cfg
):  
    bcu = []

    for tag, type, dir, val in cfg.boundary.dirichlet.velocity:
        dofs = fem.locate_dofs_topological(
            V.sub(dir), facet_tags.dim, facet_tags.find(tag)
        )

        if type == "uniform":
            bc = fem.dirichletbc(val, dofs, V.sub(dir))
            bcu.append(bc)

        elif type == "poiseuille":
            radious, center, normal = cfg.boundary.dirichlet.poiseuille
            prof = PoiseuilleProfile(
                val, radious, center, normal
            ) 
            f = fem.Function(V.sub(dir))
            f.interpolate(prof)
            bc = fem.dirichletbc(f, dofs)
            bcu.bcu.extend(bc)(bc)

        else:
            raise ValueError(f"Unknown BC type: {type}")

    return bcu

def make_pressure_bcp(
    Q: fem.FunctionSpace,
    facet_tags: MeshTags,
    cfg
):  
    bcp = []

    for tag, val in cfg.boundary.dirichlet.pressure:
        dofs = fem.locate_dofs_topological(
            Q, facet_tags.dim, facet_tags.find(tag)
        )

        bc = fem.dirichletbc(val, dofs, V=Q)
        bcp.append(bc)

    return bcp

def make_pressure_anchor(
    Q: fem.FunctionSpace,
    msh: mesh.Mesh,
    cfg
):
    tag, value = cfg.boundary.dirichlet.pressure[0]

        
    def near_anchor_point(x):
        return np.isclose(x[0], 0) & np.isclose(x[1], 0)
        
    anchor_vertices = mesh.locate_entities(
        msh, 
        dim = 0, 
        marker = near_anchor_point
    )

    dofs_anchor = fem.locate_dofs_topological(
        Q, 
        0, 
        anchor_vertices
    )
            
    bc_anchor = fem.dirichletbc(
        value = value, 
        dofs = dofs_anchor, 
        V = Q
    )

    return [bc_anchor]


class NavierStokes():
    """ N.S. eq. """

    def __init__(
        self,
        V: fem.FunctionSpace,
        Q: fem.FunctionSpace,
        Re: float,
        t_end: float,
        dt: float,
        dx,
        bcu,
        bcp,
    ):
        self.dt, self.Re = dt, Re
        self.t_end = t_end
        self.V, self.Q = V, Q
        self.bcu, self.bcp = bcu, bcp
        self.dx = dx

        self.u, self.v = ufl.TrialFunction(V), ufl.TestFunction(V)
        self.p, self.q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

        self.u_, self.p_ = fem.Function(V), fem.Function(Q)
        self.u_n, self.p_n = fem.Function(V), fem.Function(Q)

    def build_weakform(self, dx):
        """ projection matrix """
        # Step 1
        self.lhs1 = form(
            ufl.dot(self.u / self.dt, self.v) * dx
            + 0.5 * (1.0 / self.Re) 
            * ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) 
            * dx
        )

        self.rhs1 = form(
            ufl.dot(
                self.u_n / self.dt, 
                self.v
            ) * dx
            - ufl.dot(
                ufl.dot(self.u_n, ufl.nabla_grad(self.u_n)), 
                self.v
            ) * dx
            - 0.5 * (1.0 / self.Re) * 
            ufl.inner(ufl.grad(self.u_n), ufl.grad(self.v)) 
            * dx
        )

        self.A1 = petsc.assemble_matrix(self.lhs1, bcs = self.bcu)
        self.A1.assemble()
        self.b1 = petsc.create_vector(self.rhs1)
    
        # Step 1
        self.lhs2 = form(
            ufl.inner(ufl.grad(self.p), ufl.grad(self.q)) * dx 
        )
        self.rhs2 = form(
            -(1.0 / self.dt) * ufl.div(self.u_) * self.q * dx
        )
   
        self.A2 = petsc.assemble_matrix(self.lhs2, bcs = self.bcp)
        self.A2.assemble()
        self.b2 = petsc.create_vector(self.rhs2)

        # Step 2
        self.lhs3 = form(
            ufl.dot(self.u, self.v) * dx
        )
        self.rhs3 = form(
            ufl.dot(self.u_, self.v) * dx 
            - self.dt * ufl.dot(ufl.nabla_grad(self.p_), self.v) * dx
        )

        # Step 3
        self.A3 = assemble_matrix(self.lhs3)
        self.A3.assemble()
        self.b3 = create_vector(self.rhs3)

    def setup_solver_petsc(self, msh):
        """ petsc setting """
        # Solver for step 1
        self.solver1 = PETSc.KSP().create(msh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        self.pc1 = self.solver1.getPC()
        self.pc1.setType(PETSc.PC.Type.HYPRE)
        self.pc1.setHYPREType("boomeramg")

        # Solver for step 2
        self.solver2 = PETSc.KSP().create(msh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        self.pc2 = self.solver2.getPC()
        self.pc2.setType(PETSc.PC.Type.HYPRE)
        self.pc2.setHYPREType("boomeramg")

        # Solver for step 3
        self.solver3 = PETSc.KSP().create(msh.comm)
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
