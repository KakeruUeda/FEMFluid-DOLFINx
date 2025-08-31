import meshio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

msh = meshio.read("mesh/straight_tube.msh")
points = msh.points[:, :2]  

field_data = msh.field_data
fluid_id = field_data["fluid"][0]
wall_top_id = field_data["wall_top"][0]
wall_bottom_id = field_data["wall_bottom"][0]

print("fluid_id:", fluid_id)
print("wall_top_id:", wall_top_id)
print("wall_bottom_id:", wall_bottom_id)

fluid_node_ids = set()
wall_node_ids = set()

for cell_block, tag_block in zip(msh.cells, msh.cell_data["gmsh:physical"]):
    cell_type = cell_block.type
    data = cell_block.data

    if cell_type == "triangle6":
        for d, tag in zip(data, tag_block):
            if tag == fluid_id:
                fluid_node_ids.update(d)

    elif cell_type == "line3":
        for d, tag in zip(data, tag_block):
            if tag == wall_top_id:
                wall_node_ids.update(d)
            if tag == wall_bottom_id:
                wall_node_ids.update(d)

print("fluid_nodes:", len(fluid_node_ids))
print("wall_nodes:", len(wall_node_ids))

fluid_coords = points[sorted(fluid_node_ids)]
wall_coords = points[sorted(wall_node_ids)]

pd.DataFrame(fluid_coords).to_csv("PDE_X.csv", index=False, header=False)

zero_velocity = np.zeros_like(wall_coords) 
pd.DataFrame(wall_coords).to_csv("BC_X.csv", index=False, header=False)
pd.DataFrame(zero_velocity).to_csv("BC_U.csv", index=False, header=False)

plt.figure(figsize=(8, 4))
plt.scatter(fluid_coords[:, 0], fluid_coords[:, 1], s=5, color='blue', label='fluid')
plt.scatter(wall_coords[:, 0], wall_coords[:, 1], s=10, color='red', label='wall')
plt.axis("equal")
plt.legend()
plt.title("Extracted Points from .msh")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("points_plot.png", dpi=300)
