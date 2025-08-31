import h5py
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support
from scipy.interpolate import LinearNDInterpolator, CubicSpline

def parse_key(k):
    return float(k.replace('_', '.'))

def print_hdf5_structure(g, indent=0):
    for key in g.keys():
        item = g[key]
        print("  " * indent + f"- {key}: {type(item)}")
        if isinstance(item, h5py.Group):
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            print("  " * (indent + 1) + f"shape: {shape}, dtype: {dtype}")

with h5py.File("output/straight_tube_with_womersley_flow_2025-06-18_03-27-46/velocity.h5", "r") as f:
    print_hdf5_structure(f)
    keys = sorted(f["Function/f"].keys(), key=parse_key) 
    sample = f[f"Function/f/{keys[0]}"][:]
    num_points, dim = sample.shape
    num_time = len(keys)
    
    velocity_data_all = np.empty((num_time, num_points, dim), dtype=sample.dtype)

    for i, key in enumerate(keys):
        velocity_data_all[i] = f[f"Function/f/{key}"][:]

    points = f["Mesh/mesh/geometry"][:]

with h5py.File("output/straight_tube_with_womersley_flow_2025-06-18_03-27-46/pressure.h5", "r") as f:
    keys = sorted(f["Function/f"].keys(), key=parse_key)
    sample = f[f"Function/f/{keys[0]}"][:]
    num_points = sample.shape[0]
    num_time = len(keys)
    
    pressure_data_all = np.empty((num_time, num_points), dtype=sample.dtype)

    for i, key in enumerate(keys):
        pressure_data_all[i] = f[f"Function/f/{key}"][:].reshape(-1)

velocity_sampled = velocity_data_all[::2]
pressure_sampled = pressure_data_all[::2]

sampled_keys = keys[::2]

time_sampled = np.array([parse_key(k) for k in sampled_keys], dtype=np.float32) 
time_all = np.array([parse_key(k) for k in keys], dtype=np.float32) 
accel = np.empty_like(velocity_sampled)
velocity_spline = np.empty_like(velocity_data_all)

for point_idx in range(velocity_sampled.shape[1]):
    for dim_idx in range(velocity_sampled.shape[2]):
        vel_series = velocity_sampled[:, point_idx, dim_idx]
        spline = CubicSpline(time_sampled, vel_series)
        velocity_spline[:, point_idx, dim_idx] = spline(time_all)
        accel[:, point_idx, dim_idx] = spline.derivative()(time_sampled)

point_idx = 2121 
dim_idx = 0    

t_min = 27
t_max = 32

mask = (time_all >= t_min) & (time_all <= t_max)
mask2 = (time_sampled >= t_min) & (time_sampled <= t_max)

time_plot = time_all[mask]
time_plot2 = time_sampled[mask2]
original_plot = velocity_data_all[mask, point_idx, dim_idx]
sampled_plot = velocity_sampled[mask2, point_idx, dim_idx]
spline_plot = velocity_spline[mask, point_idx, dim_idx]

plt.figure(figsize=(6, 4))
plt.plot(time_plot, original_plot, label='original', marker='o', linestyle='None', color="red", markersize=7)
plt.xlabel("Time")
plt.ylabel("Velocity (x)")
plt.tight_layout()
plt.savefig("velocity_original.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(time_plot2, sampled_plot, label='sapmled', marker='x', linestyle='None', color="black", markersize=7)
plt.xlabel("Time")
plt.ylabel("Velocity (x)")
plt.tight_layout()
plt.savefig("velocity_sampled.png")
plt.close()

spline = CubicSpline(time_plot2, sampled_plot)

t_fine = np.linspace(time_plot2[0], time_plot2[-1], 200)
spline_interp = spline(t_fine)

plt.figure(figsize=(6, 4))
plt.plot(time_plot2, sampled_plot, label='sampled', marker='x', linestyle='None', color="black", markersize=7)
plt.plot(t_fine, spline_interp, label='Spline', linestyle='-', color="blue")
plt.xlabel("Time")
plt.ylabel("Velocity (x)")
plt.legend()
plt.tight_layout()
plt.savefig("velocity_sampled_with_spline.png")
plt.close()

t_target = 29.5
v_target = spline(t_target)
dvdt_target = spline.derivative()(t_target)

plt.figure(figsize=(6, 4))
plt.plot(time_plot2, sampled_plot, label='Sampled', marker='x', linestyle='None', color="black", markersize=7)
plt.plot(t_fine, spline_interp, label='Spline', linestyle='-', color="blue")

t_line = np.array([t_target - 0.2, t_target + 0.2])
v_line = v_target + dvdt_target * (t_line - t_target)
plt.plot(t_line, v_line, 'r--', label=f'Slope (t={t_target})')

plt.plot(t_target, v_target, 'ro')

plt.xlabel("Time")
plt.ylabel("Velocity (x)")
plt.legend()
plt.tight_layout()
plt.savefig("velocity_spline_with_slope.png")
plt.close()


points_2d = points[:, :2]

i = 147
velocity_snap = velocity_sampled[i]
accel_snap = accel[i]

x_min = -6
y_min = -6
x_max = 6
y_max = 6

Nx, Ny = 100, 100

x_grid = np.linspace(x_min, x_max, Nx + 1)
y_grid = np.linspace(y_min, y_max, Ny + 1)
dx = (x_max - x_min) / Nx
dy = (y_max - y_min) / Ny

xc = 0.5 * (x_grid[:-1] + x_grid[1:])
yc = 0.5 * (y_grid[:-1] + y_grid[1:])
x_centers, y_centers = np.meshgrid(xc, yc)
cell_centers = np.column_stack([x_centers.ravel(), y_centers.ravel()])


# --- Velocity ---
vx_interp = LinearNDInterpolator(points_2d, velocity_snap[:, 0], fill_value=0.0)
vy_interp = LinearNDInterpolator(points_2d, velocity_snap[:, 1], fill_value=0.0)

v_interp = np.column_stack([
    vx_interp(cell_centers),
    vy_interp(cell_centers)
])

velocity_3d = np.column_stack([v_interp, np.zeros(len(v_interp))])
velocity_mag = np.linalg.norm(v_interp, axis=1)


# --- Acceleration ---
ax_interp = LinearNDInterpolator(points_2d, accel_snap[:, 0], fill_value=0.0)
ay_interp = LinearNDInterpolator(points_2d, accel_snap[:, 1], fill_value=0.0)

a_interp = np.column_stack([
    ax_interp(cell_centers),
    ay_interp(cell_centers)
])

acceleration_3d = np.column_stack([a_interp, np.zeros(len(a_interp))])
acceleration_mag = np.linalg.norm(a_interp, axis=1)


def save_vector_components_to_vti(filename, vector_field, spacing, origin, dims, prefix="v"):
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)

    for i, comp in enumerate(['x', 'y', 'z']):
        array = numpy_support.numpy_to_vtk(vector_field[:, i], deep=True)
        array.SetName(f"{prefix}{comp}")
        image.GetCellData().AddArray(array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    writer.Write()

spacing = (dx, dy, 1.0)
origin = (x_min, y_min, 0.0)
dims = (Nx + 1, Ny + 1, 1)

save_vector_components_to_vti(
    "velocity_components.vti",
    velocity_3d,
    spacing, origin, dims,
    prefix="v"
)

save_vector_components_to_vti(
    "acceleration_components.vti",
    acceleration_3d,
    spacing, origin, dims,
    prefix="a"
)
tol = 1e-6

nonzero_mask = np.linalg.norm(v_interp, axis=1) >= tol

nonzero_centers = cell_centers[nonzero_mask]
nonzero_velocities = v_interp[nonzero_mask]
nonzero_acceleration = a_interp[nonzero_mask]

np.savetxt("DATA_X.csv", nonzero_centers, delimiter=",", fmt="%.8f")
np.savetxt("DATA_U.csv", nonzero_velocities, delimiter=",", fmt="%.8f")
np.savetxt("DATA_A.csv", nonzero_acceleration, delimiter=",", fmt="%.8f")

data_x = np.loadtxt("DATA_X.csv", delimiter=",")
data_u = np.loadtxt("DATA_U.csv", delimiter=",")
data_a = np.loadtxt("DATA_A.csv", delimiter=",")
data_a_mag = np.sqrt(data_a[:, 0]**2 + data_a[:, 1]**2)

def plot_field(x, values, fname, cmap, vmin=None, vmax=None):
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(x[:, 0], x[:, 1], c=values, cmap=cmap, s=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")

    plt.axis("equal")
    plt.box(False)

    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)

    sc.set_clim(vmin, vmax)

    cbar = plt.colorbar(
        sc, orientation="horizontal",
        fraction=0.05, pad=0.1
    )

    vmed = 0.5 * (vmin + vmax)
    cbar.set_ticks([vmin, vmed, vmax])
    cbar.set_ticklabels([f"{vmin:.4f}", f"{vmed:.4f}", f"{vmax:.4f}"])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)

    plt.savefig(fname, dpi=300)
    plt.close()

i = 147
# 0番目のスナップショットを取り出す
vi = velocity_sampled[i]  # shape: (num_points, 2)
pi = pressure_sampled[i]  # shape: (num_points,)
x = points[:, 0]
y = points[:, 1]
u = vi[:, 0]
v = vi[:, 1]
ax = accel_snap[:, 0]
ay = accel_snap[:, 1]

vel_mag = np.sqrt(u**2 + v**2)
accel_mag = np.sqrt(ax**2 + ay**2)

plot_field(points, u, "u_orig.png", "jet")
plot_field(points, v, "v_orig.png", "jet")
plot_field(points, vel_mag, "vel_mag_orig.png", "jet")
plot_field(points, ax, "ax_orig.png", "jet")
plot_field(points, ay, "ay_orig.png", "jet")
plot_field(points, accel_mag, "accel_mag_orig.png", "jet")
plot_field(points, pi, "pre_orig.png", "winter")

plot_field(data_x, data_u[:, 0], "data_ux.png", "jet")
plot_field(data_x, data_u[:, 1], "data_uy.png", "jet")
plot_field(data_x, data_a[:, 0], "data_ax.png", "jet")
plot_field(data_x, data_a[:, 1], "data_ay.png", "jet")
plot_field(data_x, data_a_mag, "data_a_mag.png", "jet")