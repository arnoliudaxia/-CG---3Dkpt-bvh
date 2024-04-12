import scipy.io
import numpy as np
from scipy.spatial.transform import Rotation as R
import math3d

mat_data = scipy.io.loadmat('save_data_AVG0.mat')
filename = 'mouse.bvh'

frame_num = 1797
link_num = 22
root_id = 5
links = [[5, 4], [4, 3], [3, 2], [3, 1], [5, 6], [6, 7], [7, 8], [4, 12], [12, 11], [11, 10], [10, 9], [4, 16], [16, 15], [15, 14], [14, 13], [5, 19], [19, 18], [18, 17], [5, 22], [22, 21], [21, 20]]
names = ['HeadL', 'HeadR', 'HeadF', 'SpineF', 'SpineB', 'TailF', 'TailM', 'TailB', 'HandL', 'WristL', 'ElbowL', 'ShoulderL', 'HandR', 'WristR', 'ElbowR', 'ShoulderR', 'FootL', 'AnkleL', 'HipL', 'FootR', 'AnkleR', 'HipR']
names_idx = {names[i]: i for i in range(link_num)}
data = [[mat_data['pred'][i, :, k] for k in range(link_num)] for i in range(frame_num)]
# fps = mat_data['fps'][0][0]
fps = 60
dirs = ['-x', 'x', 'y', 'y', '0', '-y', '-y', '-y', '-z', '-z', '-z', '-x', '-z', '-z', '-z', 'x', '-z', '-z', '-x', '-z', '-z', 'x']
print(names)

points = []
for i in [0] + [k for k in range(frame_num)]:
    point = []
    for j in range(link_num):
        pos = data[i][j].tolist()
        point.append((pos[0], -pos[2], pos[1]))
    points.append(point)
# with open('data.txt', 'w') as f:
#     f.write(f"{points}")

class Joint:
    def __init__(self, id, name, dir):
        self.id = id
        self.from_id = None
        self.name = name
        self.father = None
        self.child_list = []
        sign = (-1 if '-' in dir else 1)
        if dir == '0':
            self.dir = np.array([0, 0, 0])
        if 'x' in dir:
            self.dir = np.array([1, 0, 0]) * sign
        if 'y' in dir:
            self.dir = np.array([0, 1, 0]) * sign
        if 'z' in dir:
            self.dir = np.array([0, 0, 1]) * sign

    def __repr__(self):
        return str(id) + ' ' + self.name

def pose2euler(pose, joint, quats, eulers):
    channel = []
    
    if joint.id == root_id:
        channel.extend(list(pose[joint.id - 1]))

    order = None
    
    if joint.name == 'SpineB':
        x_dir = pose[names_idx['HipR']] - pose[names_idx['HipL']]
        y_dir = pose[names_idx['SpineF']] - pose[joint.id - 1]
        z_dir = None
        order = 'xzy'
    elif joint.name == 'SpineF_':
        x_dir = None
        y_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
        z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
        order = 'zxy'
    elif joint.name == 'HeadF_':
        x_dir = pose[names_idx['HeadR']] - pose[names_idx['HeadL']]
        y_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
        z_dir = None
        order = 'xzy'
    elif joint.name == 'HeadL_':
        x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
        y_dir = None
        z_dir = np.cross(x_dir, pose[names_idx['HeadF']] - pose[names_idx['SpineF']])
        order = 'zyx'
    elif joint.name == 'HeadR_':
        x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
        y_dir = None
        z_dir = np.cross(x_dir, pose[names_idx['HeadF']] - pose[names_idx['SpineF']])
        order = 'zyx'
    elif joint.name == 'TailF_':
        x_dir = None
        y_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
        z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
        order = 'zxy'
    elif joint.name in ['TailF', 'TailM']:
        x_dir = None
        y_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
        z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
        order = 'zxy'
    elif joint.name == 'ShoulderL_':
        x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
        y_dir = pose[names_idx['HeadF']] - pose[names_idx['SpineF']]
        z_dir = None
        order = 'xzy'
    elif joint.name in ['ShoulderL', 'ElbowL', 'WristL']:
        x_dir = pose[names_idx['SpineF']] - pose[names_idx['ShoulderL']]
        y_dir = None
        z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
        order = 'zyx'
    elif joint.name == 'ShoulderR_':
        x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
        y_dir = pose[names_idx['HeadF']] - pose[names_idx['SpineF']]
        z_dir = None
        order = 'xzy'
    elif joint.name in ['ShoulderR', 'ElbowR', 'WristR']:
        x_dir = pose[names_idx['ShoulderR']] - pose[names_idx['SpineF']]
        y_dir = None
        z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
        order = 'zyx'
    elif joint.name == 'HipL_':
        x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
        y_dir = pose[names_idx['SpineF']] - pose[names_idx['SpineB']]
        z_dir = None
        order = 'xzy'
    elif joint.name in ['HipL', 'AnkleL']:
        x_dir = pose[names_idx['SpineB']] - pose[names_idx['HipL']]
        y_dir = None
        z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
        order = 'zyx'
    elif joint.name == 'HipR_':
        x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
        y_dir = pose[names_idx['SpineF']] - pose[names_idx['SpineB']]
        z_dir = None
        order = 'xzy'
    elif joint.name in ['HipR', 'AnkleR']:
        x_dir = pose[names_idx['HipL']] - pose[names_idx['SpineB']]
        y_dir = None
        z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
        order = 'zyx'
    # else:
    #     x_dir = pose[5] - pose[8]
    #     y_dir = None
    #     z_dir = pose[3] - pose[4]
    #     order = 'zyx'
    
    if order:
        dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
        quats[joint.name] = math3d.dcm2quat(dcm)
    else:
        print(joint.name)
        quats[joint.name] = quats[joint.father.name].copy()
    
    local_quat = quats[joint.name].copy()
    if joint.father is not None:
        local_quat = math3d.quat_divide(
            q=quats[joint.name], r=quats[joint.father.name]
        )
    
    euler = math3d.quat2euler(
        q=local_quat, order='zxy'
    )
    euler = np.rad2deg(euler)
    eulers[joint.name] = euler
    channel.extend(euler)

    for child in sorted(joint.child_list, key=lambda x: x.name):
        ch, qu, eu = pose2euler(pose, child, quats, eulers)
        channel += ch
        quats.update(qu)
        eulers.update(eu)
    return channel, quats, eulers

def write_joint(f, joint, offset):
    f.write(offset + f"JOINT {joint.name}\n")
    f.write(offset + "{\n")
    pos = data[0][(joint.id - 1) if joint.from_id is None else (joint.from_id - 1)]
    father_pos = data[0][(joint.father.id - 1) if joint.father.from_id is None else (joint.father.from_id - 1)] if joint.father is not None else np.array([0, 0, 0])
    pos = np.linalg.norm(pos - father_pos) * joint.dir
    # pos = pos - father_pos
    f.write(offset + f"\tOFFSET {pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}\n")
    f.write(offset + "\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
    if len(joint.child_list) > 0:
        for n in sorted(joint.child_list, key=lambda x: x.name):
            write_joint(f, n, offset + "\t")
    else:
        f.write(offset + "\tEnd Site\n")
        f.write(offset + "\t{\n")
        f.write(offset + "\t\tOFFSET 0.00 0.00 0.00\n")
        f.write(offset + "\t}\n")
    f.write(offset + "}\n")

def write_pos(f, data, joint, i):
    if joint.father is None:
        f.write(f'{data[joint.id]["xyz"][0]}\t{data[joint.id]["xyz"][1]}\t{data[joint.id]["xyz"][2]}\t')
    else:
        f.write("\t")
    if len(joint.child_list) != 1:
        f.write('0\t0\t0')
    else:
        f.write(f'{data[joint.id]["d_angle"][0]}\t{data[joint.id]["d_angle"][1]}\t{data[joint.id]["d_angle"][2]}')
    if len(joint.child_list) > 0:
        for n in sorted(joint.child_list, key=lambda x: x.name):
            write_pos(f, data, n, i)

if __name__ == '__main__':
    ## 1. Skeleton
    rest_joint_list = {}
    for id in range(1, link_num+1):
        rest_joint_list[id] = Joint(id, names[id - 1], dirs[id - 1])
    for link in links:
        father_id = link[0]
        child_id = link[1]
        rest_joint_list[child_id].father = rest_joint_list[father_id]
        rest_joint_list[father_id].child_list.append(rest_joint_list[child_id])
    
    for id in range(1, link_num + 1):
        if len(rest_joint_list[id].child_list) > 1:
            joints = []
            for n in sorted(rest_joint_list[id].child_list, key=lambda x: x.name):
                joint = Joint(n.id + link_num, n.name + '_', dirs[n.id - 1])
                joint.from_id = id
                joint.father = rest_joint_list[id]
                n.father = joint
                joint.child_list.append(n)
                joints.append(joint)
            rest_joint_list[id].child_list = joints
    ## 2. Rest Pose
    with open(filename, "w") as f:
        f.write("HIERARCHY\n")
        f.write(f"ROOT {names[root_id - 1]}\n")
        f.write("{\n")
        f.write("\tOFFSET 0.00 0.00 0.00\n")
        f.write("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        for n in sorted(rest_joint_list[root_id].child_list, key=lambda x: x.name):
            write_joint(f, n, "\t")
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames:\t{frame_num}\n")
        f.write(f"Frame Time:\t{1. / fps:.6f}\n")

        print('hierarchy done')

        # 3. Pose for each frame
        for i in range(frame_num):
            channel, _, _ = pose2euler(data[i], rest_joint_list[root_id], {}, {})
            f.write(' '.join([f'{element}' for element in channel]) + '\n')