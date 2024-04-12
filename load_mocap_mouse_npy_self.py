import scipy.io
import numpy as np
from scipy.spatial.transform import Rotation as R
import math3d

mat_data = np.load("tri_new.npy")
filename = 'mouse-24.bvh'

frame_num = 1797
root_id = 1
links = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7,11], [8,12],[9,13],[10,14],[10,15],[10,16],[10,17],[10,18],[17,19],[18,20],[20,21],[21,22],[20,23],[23,24]] # 骨骼之间的连接关系
names = ['Hip', 'SpineH', 'Tail(base)', 'KneeL', 'KneeR', 'SpineG', 'Tail(mid)', 'AnkleL', 'AnkleR', 'SpineH', "Tail(end)" ,"HindpawL","HindpawR","Snout","EarL","EarR","ShoulderL","ShoulderR","ElbowL","ElbowR","WristL","ForepawL","WristR","WristR"] # 和上面的连接关系对应
jointIndexInNpy=["EarL","EarR","Snout","SpineH","SpineG","SpineH","Hip","Tail(base)","Tail(mid)","Tail(end)","ForepawL","WristL","ElbowL","ShoulderL","ForepawR","WristR","ElbowR","ShoulderR","HindpawL","AnkleL","KneeL","HindpawR","AnkleR","KneeR"]
fps=60


link_num = len(names)
print(f"骨骼的数量为：{link_num}")
names_idx = {jointIndexInNpy[i]: i for i in range(link_num)}
data = mat_data[:,:,:,:3] # frame,bone,1,4

class Joint:
    def __init__(self, id, name):
        self.id = id
        self.from_id = None
        self.name = name
        self.father = None
        self.child_list = []

    def __repr__(self):
        return str(id) + ' ' + self.name

def pose2euler(pose, joint, quats, eulers):
    channel = []
    
    if joint.id == root_id:
        channel.extend(list(pose[joint.id - 1]))

    order = None
    
    # if joint.name == 'Hip':
    #     x_dir = pose[names_idx['KneeR']] - pose[names_idx['KneeL']]
    #     y_dir = pose[names_idx['SpineH']] - pose[joint.id - 1]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name == 'SpineH':
    #     x_dir = None
    #     y_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
    #     z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
    #     order = 'zxy'
    # elif joint.name == 'HeadF_':
    #     x_dir = pose[names_idx['HeadR']] - pose[names_idx['HeadL']]
    #     y_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name == 'HeadL_':
    #     x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
    #     y_dir = None
    #     z_dir = np.cross(x_dir, pose[names_idx['HeadF']] - pose[names_idx['SpineH']])
    #     order = 'zyx'
    # elif joint.name == 'HeadR_':
    #     x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
    #     y_dir = None
    #     z_dir = np.cross(x_dir, pose[names_idx['HeadF']] - pose[names_idx['SpineH']])
    #     order = 'zyx'
    # elif joint.name == 'TailF_':
    #     x_dir = None
    #     y_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
    #     z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
    #     order = 'zxy'
    # elif joint.name in ['TailF', 'TailM']:
    #     x_dir = None
    #     y_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
    #     z_dir = np.cross(pose[names_idx['HipR']] - pose[names_idx['HipL']], y_dir)
    #     order = 'zxy'
    # elif joint.name == 'ShoulderL_':
    #     x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
    #     y_dir = pose[names_idx['HeadF']] - pose[names_idx['SpineH']]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name in ['ShoulderL', 'ElbowL', 'WristL']:
    #     x_dir = pose[names_idx['SpineH']] - pose[names_idx['ShoulderL']]
    #     y_dir = None
    #     z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
    #     order = 'zyx'
    # elif joint.name == 'ShoulderR_':
    #     x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
    #     y_dir = pose[names_idx['HeadF']] - pose[names_idx['SpineH']]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name in ['ShoulderR', 'ElbowR', 'WristR']:
    #     x_dir = pose[names_idx['ShoulderR']] - pose[names_idx['SpineH']]
    #     y_dir = None
    #     z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
    #     order = 'zyx'
    # elif joint.name == 'HipL_':
    #     x_dir = pose[joint.from_id - 1] - pose[joint.child_list[0].id - 1]
    #     y_dir = pose[names_idx['SpineH']] - pose[names_idx['SpineB']]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name in ['HipL', 'AnkleL']:
    #     x_dir = pose[names_idx['SpineB']] - pose[names_idx['HipL']]
    #     y_dir = None
    #     z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
    #     order = 'zyx'
    # elif joint.name == 'HipR_':
    #     x_dir = pose[joint.child_list[0].id - 1] - pose[joint.from_id - 1]
    #     y_dir = pose[names_idx['SpineH']] - pose[names_idx['SpineB']]
    #     z_dir = None
    #     order = 'xzy'
    # elif joint.name in ['HipR', 'AnkleR']:
    #     x_dir = pose[names_idx['HipL']] - pose[names_idx['SpineB']]
    #     y_dir = None
    #     z_dir = pose[joint.id - 1] - pose[joint.child_list[0].id - 1]
    #     order = 'zyx'
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
    # pos = np.linalg.norm(pos - father_pos) * joint.dir
    pos = (pos - father_pos)[0].tolist()
    # pos = pos - father_pos
    f.write(offset + f"\tOFFSET {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
    f.write(offset + "\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
    if len(joint.child_list) > 0:
        # for n in sorted(joint.child_list, key=lambda x: x.name):
        for n in joint.child_list:
            write_joint(f, n, offset + "\t")
    else:
        f.write(offset + "\tEnd Site\n")
        f.write(offset + "\t{\n")
        f.write(offset + "\t\tOFFSET 0.00 0.00 0.00\n")
        f.write(offset + "\t}\n")
    f.write(offset + "}\n")



# 计算两个向量之间的夹角（弧度）
def angle_between_vectors(v1, v2, dir):
    v1=v1.copy().reshape(3,)
    v2=v2.copy().reshape(3,)
    v1[dir]=0.0
    v2[dir]=0.0

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def pose2euler2(pose, joint,fatherID=None):
    channels=[]
    if joint.father is None:
        channels=[*channels,*pose[joint.id - 1].reshape(3,),*np.array([0.0,0.0,0.0])]
    else:
        zAngle=angle_between_vectors(pose[fatherID-1]-pose[joint.id - 1], joint.initalPose,2)
        yAngle=angle_between_vectors(pose[fatherID-1]-pose[joint.id - 1], joint.initalPose,1)
        xAngle=angle_between_vectors(pose[fatherID-1]-pose[joint.id - 1], joint.initalPose,0)
        # channels.append(np.array([xAngle,yAngle,zAngle]))
        channels=[*channels,*np.array([xAngle,yAngle,zAngle])]

    for n in joint.child_list:
        # channels.append(pose2euler2(pose, n,joint.id))
        channels=[*channels,*pose2euler2(pose, n,joint.id)]

    return channels
    

if __name__ == '__main__':
    ## 1. Skeleton
    rest_joint_list = {}
    for id in range(1, link_num+1):
        # rest_joint_list[id] = Joint(id, jointIndexInNpy[id - 1])
        rest_joint_list[id] = Joint(id, names[id - 1])
    for link in links:
        # father_id = jointIndexInNpy.index(names[link[0]-1])+1
        father_id = link[0]
        # child_id = jointIndexInNpy.index(names[link[1]-1])+1
        child_id = link[1]
        rest_joint_list[child_id].father = rest_joint_list[father_id]
        rest_joint_list[child_id].initalPose = data[0][father_id-1]-data[0][child_id-1]
        rest_joint_list[father_id].child_list.append(rest_joint_list[child_id])
    
    for id in range(1, link_num + 1):
        if len(rest_joint_list[id].child_list) > 1:
            joints = []
            for n in rest_joint_list[id].child_list:
                joint = Joint(n.id + link_num, n.name + '_')
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
        
        # for n in sorted(rest_joint_list[root_id].child_list, key=lambda x: x.name):
        for n in rest_joint_list[root_id].child_list:
            write_joint(f, n, "\t")
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames:\t{frame_num}\n")
        f.write(f"Frame Time:\t{1. / fps:.6f}\n")

        print('hierarchy done')

        # 构建init pose

        # # 3. Pose for each frame
        # for i in range(frame_num):
        for i in range(1):
            # channel, _, _ = pose2euler(data[i], rest_joint_list[root_id], {}, {})
            channel=pose2euler2(data[i], rest_joint_list[root_id])
            writting=[]
            for j in channel:
                if j<1e-1:
                    j=0
                writting.append(str(j))
            f.write(' '.join(writting))

            # f.write(' '.join([f'{element}' for element in channel]) + '\n')