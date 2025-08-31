import h5py
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator

with h5py.File("output/bend_tube_2_2025-06-14_03-53-20/velocity.h5", "r") as f:
    keys = sorted(f["Function/f"].keys())  
    velocity_data = f["Function/f/" + keys[-1]][:] 
    points = f["Mesh/mesh/geometry"][:]

points_2d = points[:, :2]
velocity_2d = velocity_data[:, :2]

x_min = -6
y_min = -6
x_max = 6
y_max = 6

Nx, Ny = 30, 30

x_grid = np.linspace(x_min, x_max, Nx + 1)
y_grid = np.linspace(y_min, y_max, Ny + 1)
dx = (x_max - x_min) / Nx
dy = (y_max - y_min) / Ny

xc = 0.5 * (x_grid[:-1] + x_grid[1:])
yc = 0.5 * (y_grid[:-1] + y_grid[1:])
x_centers, y_centers = np.meshgrid(xc, yc)
cell_centers = np.column_stack([x_centers.ravel(), y_centers.ravel()])

vx_interp = LinearNDInterpolator(points_2d, velocity_2d[:, 0], fill_value=0.0)
vy_interp = LinearNDInterpolator(points_2d, velocity_2d[:, 1], fill_value=0.0)

v_interp = np.column_stack([
    vx_interp(cell_centers),
    vy_interp(cell_centers)
])

velocity_3d = np.column_stack([v_interp, np.zeros(len(v_interp))])
velocity_mag = np.linalg.norm(v_interp, axis=1)

image = vtk.vtkImageData()
image.SetDimensions(Nx+1, Ny+1, 1)
image.SetSpacing(dx, dy, 1.0)
image.SetOrigin(x_min, y_min, 0.0)

vtk_velocity = numpy_support.numpy_to_vtk(velocity_3d, deep=True)
vtk_velocity.SetName("velocity")
image.GetCellData().AddArray(vtk_velocity)

vtk_vmag = numpy_support.numpy_to_vtk(velocity_mag, deep=True)
vtk_vmag.SetName("velocity_magnitude")
image.GetCellData().AddArray(vtk_vmag)

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("velocity_field.vti")
writer.SetInputData(image)
writer.SetDataModeToBinary()  
writer.Write()

tol = 1e-2

nonzero_mask = np.linalg.norm(v_interp, axis=1) >= tol

nonzero_centers = cell_centers[nonzero_mask]
nonzero_velocities = v_interp[nonzero_mask]

np.savetxt("DATA_X.csv", nonzero_centers, delimiter=",", fmt="%.8f")
np.savetxt("DATA_U.csv", nonzero_velocities, delimiter=",", fmt="%.8f")

data_x = np.loadtxt("DATA_X.csv", delimiter=",")


# Scatter plot
plt.figure(figsize=(5, 5))
plt.scatter(data_x[:, 0], data_x[:, 1], s=8, color='green')
plt.title("Non-zero velocity cell centers (DATA_X)", fontsize=10)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.savefig("downsampled.png", dpi=300)