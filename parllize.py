import subprocess

sharedArgs=['/home/lumiani/miniconda3/envs/mouseMotion_3dkpt2bvh/bin/python', 'load_mocap_mouse_npy_onlychange_parallel.py','-u']

fileList=[
    "/mnt/Data-Mouse/batch/20231121/231121_C3M1-2/tri_filter.npy",
    "/mnt/Data-Mouse/batch/20231121/231121_C3M2-2/tri_filter.npy",
]
processPool=[]

for file in fileList:
    processPool.append(subprocess.Popen([*sharedArgs,file]))


for process in processPool:
    process.wait()


  