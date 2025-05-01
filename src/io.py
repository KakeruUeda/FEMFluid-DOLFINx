from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.io import XDMFFile
from dolfinx.fem import (Function, functionspace)
from basix.ufl import element

def open_files_xdfm():
    ufile = XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w")
    pfile = XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w")

    return [ufile, pfile]

def get_vars_visu(domain):
    V_vis = element(
        "Lagrange",
        domain.msh.basix_cell(),
        domain.deg_u,
        shape = (domain.msh.geometry.dim,),
    )
    Q_vis = element(
        "Lagrange",
        domain.msh.basix_cell(),
        domain.deg_u,
    )
    u_vis = Function(functionspace(domain.msh, V_vis))
    p_vis = Function(functionspace(domain.msh, Q_vis))

    return [u_vis, p_vis]

def output_velocity(u_, u_vis, ufile, t):
    u_vis.interpolate(u_)
    ufile.write_function(u_vis, t)

def output_pressure(p_, p_vis, pfile, t):
    p_vis.interpolate(p_)
    pfile.write_function(p_vis, t)