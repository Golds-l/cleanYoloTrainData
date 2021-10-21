import glob
import os
import shutil


def makeLabelFile(imgPath):
    imgPaths = glob.glob(imgPath)
    labelPath = "".join(imgPaths[0].split("\\")[:-2])+"\\labels"
    os.mkdir(labelPath)
    for imgPt in imgPaths[:2]:
        name = labelPath + imgPt.split("\\")[-1].split(".")[0] + ".txt"
        with open(name, "w") as f:
            f.write("")


def chooseImgs(imgPath, stride):
    imgPaths = glob.glob(imgPath)
    imgSavePath = "".join(imgPaths[0].split("\\")[:-2]) + "\\saveImg"
    if not os.path.exists(imgSavePath):
        os.mkdir(imgSavePath)
    for imgNum in range(len(imgPaths)):
        if imgNum % stride == 0:
            shutil.copyfile(imgPaths[imgNum], imgSavePath+"\\"+imgPaths[imgNum].split("\\")[-1])


if __name__ == "__main__":
    makeLabelFile("a.txt")
