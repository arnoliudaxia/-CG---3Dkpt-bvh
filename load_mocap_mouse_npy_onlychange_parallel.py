import os
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

import math3d

root_id = "Hip"
jointIndexInNpy = ["EarL", "EarR", "Snout", "SpineF", "SpineG", "SpineH", "Hip", "Tail(base)", "Tail(mid)", "Tail(end)",
                   "ForepawL", "WristL", "ElbowL", "ShoulderL", "ForepawR", "WristR", "ElbowR", "ShoulderR", "HindpawL",
                   "AnkleL", "KneeL", "HindpawR", "AnkleR", "KneeR"]  # 在npy数据顺序的骨骼
JointStructure = {
    'Hip': {
        'parent': None,
        'children': ['SpineH', 'Tail(base)', 'KneeL', 'KneeR'],
        "dir": "0",
    },
    'SpineH': {
        'parent': 'Hip',
        'children': ['SpineG'],
        "dir": "y",
    },
    'Tail(base)': {
        'parent': 'Hip',
        'children': ['Tail(mid)'],
        "dir": "-y",
    },
    "KneeL": {
        'parent': 'Hip',
        'children': ["AnkleL"],
        "dir": "x",
    },
    "KneeR": {
        'parent': 'Hip',
        'children': ["AnkleR"],
        "dir": "-x",
    },
    "SpineG": {
        'parent': 'SpineH',
        'children': ["SpineF"],
        "dir": "y",
    },
    "Tail(mid)": {
        'parent': 'Tail(base)',
        'children': ["Tail(end)"],
        "dir": "-y",
    },
    "AnkleL": {
        'parent': "KneeL",
        'children': ["HindpawL"],
        "dir": "z",
    },
    "AnkleR": {
        'parent': "KneeR",
        'children': ["HindpawR"],
        "dir": "z",
    },
    "Tail(end)": {
        'parent': "Tail(mid)",
        'children': [],
        "dir": "-y",
    },
    "HindpawL": {
        'parent': "AnkleL",
        'children': [],
        "dir": "z",
    },
    "HindpawR": {
        'parent': "AnkleR",
        'children': [],
        "dir": "z",
    },
    "Snout": {
        'parent': "SpineF",
        'children': [],
        "dir": "y",
    },
    "EarL": {
        'parent': "SpineF",
        'children': [],
        "dir": "-z",
    },
    "EarR": {
        'parent': "SpineF",
        'children': [],
        "dir": "-z",
    },
    "ShoulderL": {
        'parent': "SpineF",
        'children': ["ElbowL"],
        "dir": "x",
    },
    "ShoulderR": {
        'parent': "SpineF",
        'children': ["ElbowR"],
        "dir": "-x",
    },
    "ElbowL": {
        'parent': "ShoulderL",
        'children': ["WristL"],
        "dir": "z",
    },
    "ElbowR": {
        'parent': "ShoulderR",
        'children': ["WristR"],
        "dir": "z",
    },
    "WristL": {
        'parent': "ElbowL",
        'children': ["ForepawL"],
        "dir": "z",
    },
    "WristR": {
        'parent': "ElbowR",
        'children': ["ForepawR"],
        "dir": "z",
    },
    "ForepawL": {
        'parent': "WristL",
        'children': [],
        "dir": "y",
    },
    "ForepawR": {
        'parent': "WristR",
        'children': [],
        "dir": "y",
    },
    "SpineF": {
        'parent': "SpineG",
        'children': ["Snout","EarL", "EarR", "ShoulderL", "ShoulderR"],
        "dir": "y",
    },

}
print(f"骨骼的数量为：{len(JointStructure)}")

for key in JointStructure.keys():
    JointStructure[key]["dataindex"] = jointIndexInNpy.index(key)

fps = 60

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
        return str(self.id) + ' ' + self.name

def getIndexOfJointInNpy(jointName:str):
    return JointStructure[jointName]["dataindex"]


def pose2euler(pose, jointName, quats, eulers):
    """

    :param pose:一帧的完整数据
    """
    channel = []
    thisJoint=JointStructure[jointName]
    thisJointIndex = thisJoint["dataindex"]
    fatherJointName=thisJoint["parent"]

    if jointName == root_id:
        channel.extend(list(pose[thisJointIndex]))

    order = None

    # y_dir=-y_dir
    # x_dir=-x_dir

    # 关键：上面的dir定义要定义好当前节点指向父节点的方法向量；而这里确保x_dir，y_dir，z_dir组成的向量符合世界坐标系的global方向！！

    if jointName in ['Hip'] or "Spine" in jointName:
        x_dir = pose[getIndexOfJointInNpy('KneeL')] - pose[getIndexOfJointInNpy('KneeR')]
        z_dir = None
    if jointName == 'Hip':
        y_dir = pose[thisJointIndex]-pose[getIndexOfJointInNpy('SpineH')]
        order = 'xzy'
    if jointName == 'SpineH':
        y_dir = pose[thisJointIndex]-pose[getIndexOfJointInNpy('SpineG')]
        z_dir = None
        order = 'xzy'
    if jointName == 'SpineG':
        y_dir = pose[thisJointIndex]-pose[getIndexOfJointInNpy('SpineF')]
        order = 'xzy'
    if jointName == 'SpineF':
        x_dir = pose[getIndexOfJointInNpy('ShoulderL')] - pose[getIndexOfJointInNpy('ShoulderR')]
        y_dir = pose[getIndexOfJointInNpy('Snout')] - pose[thisJointIndex]
        y_dir=-y_dir
        order = 'xzy'
    elif jointName == 'Snout':
        x_dir = pose[getIndexOfJointInNpy('EarR')] - pose[getIndexOfJointInNpy('EarL')]
        y_dir = pose[getIndexOfJointInNpy("SpineF")] - pose[thisJointIndex]
        # y_dir=-y_dir
        z_dir = None
        order = 'xzy'
    elif jointName == 'EarL':
        x_dir = pose[thisJointIndex] - pose[getIndexOfJointInNpy('EarR')]
        y_dir = None
        z_dir = np.cross(x_dir, pose[getIndexOfJointInNpy('Snout')] - pose[getIndexOfJointInNpy('SpineF')])
        order = 'zyx'
    elif jointName == 'EarR':
        x_dir = pose[thisJointIndex] - pose[getIndexOfJointInNpy('EarL')]
        y_dir = None
        z_dir = np.cross(x_dir, pose[getIndexOfJointInNpy('Snout')] - pose[getIndexOfJointInNpy('SpineF')])
        order = 'zyx'

    elif jointName == 'Tail(base)':
        x_dir = None
        y_dir =  pose[getIndexOfJointInNpy('Hip')] - pose[thisJointIndex]
        y_dir=-y_dir
        z_dir = np.cross(pose[getIndexOfJointInNpy('KneeR')] - pose[getIndexOfJointInNpy('KneeL')], y_dir)
        order = 'zxy'
    elif jointName == 'Tail(mid)':
        x_dir = None
        y_dir = pose[getIndexOfJointInNpy('Tail(base)')]-pose[thisJointIndex]
        y_dir=-y_dir
        z_dir = np.cross(pose[getIndexOfJointInNpy('KneeR')] - pose[getIndexOfJointInNpy('KneeL')], y_dir)
        order = 'zxy'

    elif jointName == 'ShoulderL':
        x_dir = pose[getIndexOfJointInNpy('ShoulderL')] - pose[getIndexOfJointInNpy('ShoulderR')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('ShoulderL')] - pose[getIndexOfJointInNpy('ElbowL')]
        order = 'zyx'
    elif jointName == 'ShoulderR':
        x_dir = pose[getIndexOfJointInNpy('ShoulderR')] - pose[getIndexOfJointInNpy('ShoulderL')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('ShoulderR')] - pose[getIndexOfJointInNpy('ElbowR')]
        order = 'zyx'

    elif jointName == 'ElbowL':
        x_dir = pose[getIndexOfJointInNpy('ElbowL')] - pose[getIndexOfJointInNpy('ElbowR')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('ElbowL')] - pose[getIndexOfJointInNpy('ForepawL')]
        order = 'zyx'
    elif jointName == 'ElbowR':
        x_dir = pose[getIndexOfJointInNpy('ElbowR')] - pose[getIndexOfJointInNpy('ElbowL')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('ElbowR')] - pose[getIndexOfJointInNpy('ForepawR')]
        order = 'zyx'

    elif jointName == 'KneeL':
        x_dir = pose[getIndexOfJointInNpy('KneeL')] - pose[getIndexOfJointInNpy('KneeR')]
        y_dir = pose[getIndexOfJointInNpy('KneeL')] - pose[getIndexOfJointInNpy('AnkleL')]
        y_dir=-y_dir
        z_dir = None
        order = 'xzy'
    elif jointName == 'KneeR':
        x_dir = pose[getIndexOfJointInNpy('KneeR')] - pose[getIndexOfJointInNpy('KneeL')]
        y_dir = pose[getIndexOfJointInNpy('KneeR')] - pose[getIndexOfJointInNpy('AnkleR')]
        z_dir = None
        order = 'xzy'
    elif jointName == 'AnkleL':
        x_dir =  pose[getIndexOfJointInNpy('AnkleL')] - pose[getIndexOfJointInNpy('AnkleR')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('AnkleL')] - pose[getIndexOfJointInNpy('KneeL')]
        x_dir=-x_dir
        order = 'zyx'
    elif jointName == 'AnkleR':
        x_dir =  pose[getIndexOfJointInNpy('AnkleL')] - pose[getIndexOfJointInNpy('AnkleR')]
        y_dir = None
        z_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[thisJointIndex]
        # z_dir=-z_dir
        # x_dir=-x_dir
        order = 'zyx'

    ###############处理末端节点###############

    elif jointName=="HindpawL" or jointName=="HindpawR":
        x_dir =  pose[getIndexOfJointInNpy('HindpawR')] - pose[getIndexOfJointInNpy('HindpawL')]
        y_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[getIndexOfJointInNpy('Snout')]
        z_dir = None
        order = 'xzy'

    elif jointName == 'Snout':
        x_dir =  pose[getIndexOfJointInNpy('EarR')] - pose[getIndexOfJointInNpy('EarL')]
        y_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[getIndexOfJointInNpy('Snout')]
        z_dir = None
        order = 'xzy'
    
    elif jointName == 'EarL' or jointName == 'EarR':
        x_dir =  pose[getIndexOfJointInNpy('EarR')] - pose[getIndexOfJointInNpy('EarL')]
        y_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[getIndexOfJointInNpy('Snout')]
        z_dir = None
        order = 'xzy'

        # x_dir =  pose[getIndexOfJointInNpy('EarR')] - pose[getIndexOfJointInNpy('EarL')]
        # # y_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[getIndexOfJointInNpy('Snout')]
        # y_dir = None
        # z_dir = pose[getIndexOfJointInNpy('SpineF')] - pose[getIndexOfJointInNpy('Snout')]
        # # order = 'xzy'
        # order = 'zyx'
    elif jointName == 'ForepawL' or jointName == 'ForepawR':
        x_dir =  pose[getIndexOfJointInNpy('ForepawR')] - pose[getIndexOfJointInNpy('ForepawL')]
        y_dir = pose[getIndexOfJointInNpy('SpineG')] - pose[getIndexOfJointInNpy('SpineF')]
        z_dir = None
        order = 'xzy'
        
    elif jointName == 'Tail(end)':
        x_dir = None
        y_dir = pose[getIndexOfJointInNpy('Tail(mid)')]-pose[thisJointIndex]
        y_dir=-y_dir
        z_dir = np.cross(pose[getIndexOfJointInNpy('KneeR')] - pose[getIndexOfJointInNpy('KneeL')], y_dir)
        order = 'zxy'
    


    if order:
        dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)
        quats[jointName] = math3d.dcm2quat(dcm)
    else:
        # print(jointName)
        quats[jointName] = quats[fatherJointName].copy()

    local_quat = quats[jointName].copy()
    if fatherJointName is not None:
        local_quat = math3d.quat_divide(
            q=quats[jointName], r=quats[fatherJointName]
        )

    euler = math3d.quat2euler(
        q=local_quat, order='zxy'
    )
    euler = np.rad2deg(euler)
    eulers[jointName] = euler
    channel.extend(euler)

    for child in thisJoint["children"]:
        ch, qu, eu = pose2euler(pose, child, quats, eulers)
        channel += ch
        quats.update(qu)
        eulers.update(eu)
    return channel, quats, eulers


def dir2vec(dir:str):
    sign = (-1 if '-' in dir else 1)
    if dir == '0':
        return np.array([0, 0, 0])
    if 'x' in dir:
        return np.array([1, 0, 0]) * sign
    if 'y' in dir:
        return np.array([0, 1, 0]) * sign
    if 'z' in dir:
        return np.array([0, 0, 1]) * sign
def write_joint(f, jointName:str, offset):
    f.write(offset + f"JOINT {jointName}\n")
    f.write(offset + "{\n")
    
    thisJoint= JointStructure[jointName]
    fatherJoint=JointStructure[thisJoint["parent"]]

    pos=data[0][thisJoint["dataindex"]]
    father_pos = np.array([0, 0, 0])
    if fatherJoint is not None:
        father_pos = data[0][fatherJoint["dataindex"]]

    pos = np.linalg.norm(pos - father_pos) * dir2vec(thisJoint["dir"]) * 100
    # pos = pos - father_pos
    if len(thisJoint["children"])==0:
        f.write(offset + "\tOFFSET 0.00 0.00 0.00\n")
        f.write(offset + "\tCHANNELS 3 Zrotation Xrotation Yrotation\n")
        f.write(offset + "\tEnd Site\n")
        f.write(offset + "\t{\n")
        f.write(offset + f"\t\tOFFSET {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\n")
        f.write(offset + "\t}\n")
    else:
        f.write(offset + f"\tOFFSET {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\n")
        f.write(offset + "\tCHANNELS 3 Zrotation Xrotation Yrotation\n")

        for n in thisJoint["children"]:
            write_joint(f, n, offset + "\t")


    f.write(offset + "}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    这个程序可以把3d kpts的信息（tri_new.npy文件）转换为bvh文件，这样可以导入到Deephase的unity工程中导出deepphase信息\n
    使用方法：\n
    python load_mocap_mouse_npy_onlychange_parallel.py -u 数据文件路径\n
    注意「数据文件路径」所在文件夹的名字应该是可以区分的名字，因为会用在输出文件名中\n
    \n
    输出文件在「OutputBvhs/」中
    """)
    parser.add_argument('--url',"-u", type=str, help="数据文件的完整路径（tri_new.npy）")
    # 从命令行中结构化解析参数
    args = parser.parse_args()
    dataPath = Path(args.url)

    label = dataPath.parts[-2]
    data = np.load(str(dataPath))
    filename = f'OutputBvhs/mouse-24-{label}.bvh'
    # data = data[:,:,:,:3] # frame,bone,1,4
    # data=np.squeeze(data)
    # data=data[0:300]
    # np.save("fore300.npy",data)

    frame_num = data.shape[0]


    os.makedirs("OutputBvhs/",exist_ok=True)
    ## 2. Rest Pose
    with open(filename, "w") as f:
        f.write("HIERARCHY\n")
        f.write(f"ROOT {root_id}\n")
        f.write("{\n")
        f.write("\tOFFSET 0.00 0.00 0.00\n")
        f.write("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        for n in JointStructure[root_id]["children"]:
            write_joint(f, n, "\t")
        f.write("}\n")
        f.write("MOTION\n")
        f.write(f"Frames:\t{frame_num}\n")
        f.write(f"Frame Time:\t{1. / fps:.6f}\n")

        print('hierarchy done')
        # exit(0)

        # 3. Pose for each frame
        # for i in tqdm(range(frame_num)):
        for i in tqdm(range(50)):
            channel, _, _ = pose2euler(data[i], root_id, {}, {})
            f.write(' '.join([f'{element}' for element in channel]) + '\n')
