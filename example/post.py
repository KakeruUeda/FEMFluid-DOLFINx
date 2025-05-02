import h5py
import numpy as np
import vtk
from vtk.util import numpy_support


with h5py.File("output/straight_tube_2025-05-02_07-09-26/velocity.h5", "r") as f:
    velocity_data = f["Function/f/0"][:]  
    points = f["Mesh/mesh/geometry"][:]  

points_2d = points[:, :2]
velocity_2d = velocity_data[:, :2]

x_min = 0
y_min = 0
x_max = 2
y_max = 2

Nx, Ny = 20, 20

x_grid = np.linspace(x_min, x_max, Nx + 1)
y_grid = np.linspace(y_min, y_max, Ny + 1)
dx = (x_max - x_min) / Nx
dy = (y_max - y_min) / Ny

xc = 0.5 * (x_grid[:-1] + x_grid[1:])
yc = 0.5 * (y_grid[:-1] + y_grid[1:])
x_centers, y_centers = np.meshgrid(xc, yc)
cell_centers = np.column_stack([x_centers.ravel(), y_centers.ravel()])

from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
vx_interp = LinearNDInterpolator(points_2d, velocity_2d[:, 0], fill_value=0.0)
vy_interp = LinearNDInterpolator(points_2d, velocity_2d[:, 1], fill_value=0.0)

v_interp = np.column_stack([
    vx_interp(cell_centers),
    vy_interp(cell_centers)
])

print("v_interp.shape =", v_interp.shape)

velocity_3d = np.column_stack([v_interp, np.zeros(len(v_interp))])
velocity_mag = np.linalg.norm(v_interp, axis=1)

image = vtk.vtkImageData()
image.SetDimensions(Nx+1, Ny+1, 1)
image.SetSpacing(dx, dy, 1.0)
image.SetOrigin(x_min + dx / 2, y_min + dy / 2, 0.0)

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

