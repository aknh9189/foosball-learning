import glob, os

mesh_dir = "assets/foosball_table/meshes"

color_team_1 = "Kd 0.8 0.2 0.2"
color_team_2 = "Kd 0.2 0.2 0.8"

mesh_files = sorted(glob.glob(os.path.join(mesh_dir,"*.obj")))
mtl_files = sorted(glob.glob(os.path.join(mesh_dir,"*.mtl")))
print(len(mesh_files))
for mesh_file, mtl_file in zip(mesh_files, mtl_files):
    if "centered" not in mesh_file or "team" in mesh_file:
        continue
    for i, color_name in zip([1, 2], [color_team_1, color_team_2]):
        # generate mesh files for team
        with open(mtl_file, "r") as f:
            mtl = f.read()
        mtl = mtl.replace("Kd 0.627451 0.627451 0.627451", color_name)
        new_filename = mtl_file.replace(".mtl", f"_team_{i}.mtl")
        with open(new_filename, "w") as f:
            f.write(mtl)
        
        # write mesh files for team
        with open(mesh_file, "r") as f:
            mesh = f.read()
        mesh = mesh.replace(f"mtllib {os.path.basename(mtl_file)}", f"mtllib {os.path.basename(new_filename)}") 
        new_filename = mesh_file.replace(".obj", f"_team_{i}.obj")
        with open(new_filename, "w") as f:
            f.write(mesh)
        
