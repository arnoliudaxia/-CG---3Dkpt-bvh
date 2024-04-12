import bpy
names = ['HeadL', 'HeadR', 'HeadF', 'SpineF', 'SpineB', 'TailF', 'TailM', 'TailB', 'HandL', 'WristL', 'ElbowL', 'ShoulderL', 'HandR', 'WristR', 'ElbowR', 'ShoulderR', 'FootL', 'AnkleL', 'HipL', 'FootR', 'AnkleR', 'HipR']

def create_animation(points_per_frame):
    # Clear all existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    meshes = []

    # For each point, create a sphere at the initial location
    for i in range(len(points_per_frame[0])):
        initial_location=points_per_frame[0][i]
    #for initial_location in points_per_frame[0]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=initial_location)
        obj = bpy.context.active_object
        obj.name = names[i]  # 设置对象的名称
        meshes.append(obj)

    # For each frame and each point, set the location of the mesh
    for frame, locations in enumerate(points_per_frame):
        for obj, location in zip(meshes, locations):
            obj.location = location
            obj.keyframe_insert(data_path="location", frame=frame)

# Example usage:

# This is an example where there are 2 points and 3 frames
# The format is: points_per_frame[frame][point]
points = []
with open(r'D:\下载\Send_1699850829373\data.txt', 'r') as f:
    points = eval(f.read())

create_animation(points)
