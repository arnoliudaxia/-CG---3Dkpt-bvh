import os



with open("mouse.bvh","r") as f:
    lines=f.readlines()

    isStartFrames=False
    for line in lines:
        if "Frame Time" in line:
            isStartFrames=True
            continue
        if isStartFrames:
            framesNum=len(line.split(" "))
            if framesNum!=96:
                print(framesNum)