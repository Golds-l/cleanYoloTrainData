import os
import glob
import shutil
from threading import Thread
import cv2
from tqdm import tqdm
import math
import numpy as np

# Stu IDs
stuNums = ["0", "1"]

stuIdDic = {"test": "test"}


# rm illegal str or char in file names /去除文件名中的某个字符串或字符
def rename(path):
    """
    :param path: a glob path
    :return: None
    """
    pathList = glob.glob(path)
    for pt in pathList:
        print(pt)
        os.rename(pt, pt.replace(".xml", ""))


# rm mismatch labels /去除多余标签
def removeMismatchLabels(imgPath, lblPath):
    """
    :param imgPath: a glob path. example: "bigcoalSecond\\images\\val\\*.jpg"
    :param lblPath: a glob path. example: "bigcoalSecond\\images\\val\\*.txt"
    :return: num of mismatch labels
    """
    numRemove = 0
    imgPaths = glob.glob(imgPath)
    lblPaths = glob.glob(lblPath)
    imgName = [i.split("\\")[-1].split(".")[0] for i in imgPaths]  # .split("\\")[-1].split(".")[0]
    lblName = [i.split("\\")[-1].split(".")[0] for i in lblPaths]
    labelRootPath = "\\".join(imgPath.split("\\")[:-1])
    for iName in lblName:
        if iName not in imgName:
            numRemove += 1
            os.remove(f"{labelRootPath}\\{iName}.txt")  # 此处需要更改!
            print(iName)
    return numRemove


# find illegal labels /去除非法标签
def findIllegalLabels(path):
    """
    :param path: a glob path
    :return: None
    """
    pathList = glob.glob(path)
    legalLbl = ["0", "1"]
    for ptTxt in pathList:
        with open(ptTxt, "r") as T:
            dataTxt = T.readlines()
            for line in dataTxt:
                line = line.split(" ")
                if line[0] not in legalLbl:
                    print(ptTxt)
                    print(line)


# find a label file and print it /找到某个标签文件并打印
def findLabelFile(labelName, goal):
    """
    :param labelName: str, a label name. example: "20201213191353mp4@191.txt"
    :param goal: a glob path. example: "Data\\bigcoalSecond\\labels\\*.txt"
    :return: None
    """
    files = glob.iglob(goal)
    for file in files:
        if labelName == file.split("\\")[-1]:
            print(file)


# find mismatch in two list /找到两个文件夹里不匹配的文件
def match(pathI, pathII):
    """
    :param pathI: a glob path. example: "E:\\bigcoalFirst\\images\\train\\*.jpg"
    :param pathII: a glob path. example: "E:\\bigcoalSecond\\images\\train\\*.jpg"
    :return: None
    """
    pathF = glob.glob(pathI)
    pathS = glob.glob(pathII)
    mismatchList = []
    for i in pathF:
        if i not in pathS:
            mismatchList.append(i)
    print(len(mismatchList))
    print(mismatchList)


# find image without label /找到没有标签的图片
def findImgWithoutLbl(pathI, pathII):
    """
    :param pathI: a glob path. example: "D:\\大块煤数据/大块煤第三次标注数据\\images\\*.jpg"
    :param pathII: a glob path. example: "D:\\大块煤数据/大块煤第三次标注数据\\labels\\*.txt"
    :return: num of image which not has label
    """
    num = 0
    pathI = glob.glob(pathI)
    pathII = glob.glob(pathII)
    imgNames = [i.split("\\")[-1].split(".")[0] for i in pathI]
    lblNames = [i.split("\\")[-1].split(".")[0] for i in pathII]
    for img in imgNames:
        if img not in lblNames:
            print(img)
            num += 1
    return num


# find label without image /找到无对应图片的标签
def findLblWithoutImg(pathI, pathII):
    """
    :param pathI: a glob path. example: "D:/大块煤数据/大块煤第三次标注数据/images/*.jpg"
    :param pathII: a glob path. example: "D:/大块煤数据/大块煤第三次标注数据/labels/*.txt"
    :return: num of image which not has label
    """
    num = 0
    pathI = glob.glob(pathI)
    pathII = glob.glob(pathII)
    imgNames = [i.split("\\")[-1].split(".")[0] for i in pathI]
    lblNames = [i.split("\\")[-1].split(".")[0] for i in pathII]
    for lbl in lblNames:
        if lbl not in imgNames:
            print(lbl)
            num += 1
    return num


# divide train dataset and val dataset /划分YOLO训练集和验证集
def divide(imagePath, labelPath, stride, savePath):
    """
    :param savePath: path\\to\\save
    :param imagePath: a glob path
    :param labelPath: a glob path, too
    :param stride:
    :return: num of val
    """
    if savePath.endswith("\\"):
        savePath = savePath[:-1]
    os.mkdir(savePath + "\\" + "images")
    os.mkdir(savePath + "\\" + "labels")
    os.mkdir(savePath + "\\" + "images\\" + "train")
    os.mkdir(savePath + "\\" + "images\\" + "val")
    os.mkdir(savePath + "\\" + "labels\\" + "train")
    os.mkdir(savePath + "\\" + "labels\\" + "val")
    numVal = 0
    imgPaths = glob.glob(imagePath)
    labelPaths = glob.glob(labelPath)
    os.mkdir(savePath + "")
    for pathNum in tqdm(range(len(imgPaths))):
        if pathNum % stride == 0:
            shutil.copyfile(imgPaths[pathNum],
                            f"{savePath}\\images\\val\\" + imgPaths[pathNum].split("\\")[-1])
            shutil.copyfile(labelPaths[pathNum],
                            f"{savePath}\\labels\\val\\" + labelPaths[pathNum].split("\\")[-1])
            numVal += 1
        else:
            shutil.copyfile(imgPaths[pathNum],
                            f"{savePath}\\images\\train\\" + imgPaths[pathNum].split("\\")[-1])
            shutil.copyfile(labelPaths[pathNum],
                            f"{savePath}\\labels\\train\\" + labelPaths[pathNum].split("\\")[-1])
    return numVal


# rename files by number /将文件以序号重命名
def renameVideo(videoFilePath):
    """
    :param videoFilePath: a glob path
    :return: None
    """
    videoPath = glob.glob(videoFilePath)
    for i in range(len(videoPath)):
        try:
            name = str(i) + "." + videoPath[i].split(".")[-1]
            os.rename(videoPath[i], "\\".join(videoPath[i].split("\\")[:-1]) + "\\" + name)
            print("\\".join(videoPath[i].split("\\")[:-1]) + "\\" + name)
        except:
            continue
    print("end")


# make images from video /视频转图片序列
def videoToimages(videoPath, savePath, stride):
    """
    :param videoPath: path\\to\\video
    :param savePath: path\\to\\save
    :param stride: split width
    :return: None
    """
    video = cv2.VideoCapture(videoPath)
    num = 0
    name = videoPath.split("\\")[-1].split(".")[0]
    ret, firstFrame = video.read()
    while ret:
        ret, frame = video.read()
        if num % stride == 0 and ret:
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080))
            cv2.imwrite(savePath + f"\\{name}_{num}" + ".jpg", frame)
            print(savePath + f"\\{name}_{num}" + ".jpg")
        num += 1
    print("end")


# split files /将文件分配到不同文件夹内
def divideToFolders(filePath, savePath):
    """
    :param filePath: path\\to\\files
    :param savePath: path\\to\\save\\files
    :return: None
    """
    paths = glob.glob(filePath)
    for member in stuIdDic.keys():
        os.mkdir(f"{savePath}\\{member}")
    length = math.ceil(len(paths) / len(stuIdDic.keys()))
    # print(len(paths) / len(stuIdDic.keys()))
    for n in range(len(stuIdDic.keys())):
        beginIndex = n * length
        endIndex = (n + 1) * length
        if endIndex > len(paths):
            endIndex = len(paths)
        for i in paths[beginIndex:endIndex]:
            fileName = i.split("\\")[-1]
            shutil.copyfile(i, f"{savePath}\\{list(stuIdDic.keys())[n]}\\{fileName}")


# Are two images same? /判断两张图片是否相同
def isSameImg(imgPaths):
    """
    :param imgPaths: a glob path
    :return:
    """
    paths = glob.glob(imgPaths)
    paths.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    for ptNum in range(len(paths))[:-1]:
        absImg = cv2.absdiff(cv2.imread(paths[ptNum + 1]), cv2.imread(paths[ptNum]))
        gray = cv2.cvtColor(absImg, cv2.COLOR_BGR2GRAY)
        ret, binaryImg = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binaryImg) == 0:
            print(paths[ptNum])


# 标签转换 VOC转YOLO
def vocToYolo():
    pass


# rotate images(180°) and labels /旋转图片180°并更改标签
def rotateImgAndLbl(imgPath, lblPath):
    """
    :param imgPath: a glob path
    :param lblPath: a glob path, too
    :return:
    """
    imgPaths = glob.glob(imgPath)
    for imgPt in imgPaths[:5]:
        image = cv2.imread(imgPt)
        lblName = imgPt.split("\\")[-1].split(".")[0]
        lblPt = lblPath + "\\" + lblName + ".txt"
        lblsRotateCon = []
        with open(lblPt, "r", encoding="utf-8") as lbl:
            lblCon = lbl.readlines()
            for lblNum in range(len(lblCon)):
                coordinates = lblCon[lblNum].replace("\n", "").split(" ")
                coordinatesRotate = [str(1 - float(coordinates[coorNum])) if coorNum == 1 or coorNum == 2 else coordinates[coorNum] for coorNum in range(len(coordinates))]
                rotateCon = " ".join(coordinatesRotate)
                lblsRotateCon.append(rotateCon)
        lblRotatePt = lblPath + "\\" + lblName + "_rotate.txt"
        with open(lblRotatePt, "w", encoding="utf-8") as lblRotate:
            for lblRotateCon in lblsRotateCon:
                lblRotate.write(lblRotateCon)
                lblRotate.write("\n")
        cv2.imwrite("\\".join(imgPt.split("\\")[:-1]) + "\\" + imgPt.split("\\")[-1].split(".")[0] + "_rotate.jpg",
                    cv2.rotate(image, cv2.ROTATE_180))


if __name__ == "__main__":
    # isSameImg("testImgs\\*.jpg")
    rotateImgAndLbl("D:\\大块煤数据\\大块煤第三次标注图片标签数据\\*.jpg", "testImgs")
    # cv2.imshow("origin", cv2.imread("testImgs/testLbl.jpg"))
    # cv2.imshow("test", cv2.imread("testImgs/testLbl.jpg")[328: 400, 118: 148])
    # cv2.waitKey(0)
    # divideToFolders(".\\testImgs\\*.jpg", ".\\")
    # paths = glob.glob("D:\\大块煤数据\\大块煤末次标注视频\\*")
    # paths.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    # for i in paths[23:]:
    #     videoToimages(i, "D:\\bigcoalLastImg")
    # videoToimages("D:\\大块煤数据\\大块煤末次标注视频\\4.mp4", "D:\\bigcoalLastImg")
    # threads = []
    # for i in glob.glob("D:\\大块煤数据\\大块煤末次标注视频\\*")[:2]:
    #     threads.append(Thread(target=videoToimages, args=(i, "D:\\bigcoalLastImg")))
    # for i in threads:
    #     i.start()
    # for i in glob.glob("D:\\bigcoalLastImg\\*.jpg"):
    #     os.remove(i)
    #     print(i)
    # renameVideo("D:\\大块煤数据\\大块煤末次标注视频 - 副本\\*.mp4")
    # print(findLblWithoutImg("D:/大块煤数据/大块煤第三次标注数据/images/*.jpg", "D:/大块煤数据/大块煤第三次标注数据/labels/*.txt"))
    # rename("D:\\Projects\\testdata\\*.txt")
    # print(divide(6))
    # print(remove()) ..\yolov5\data\bigcoalSecond\labels\train\20201213191353mp4@191.txt
    # cleanData("D:\\bigCoalTrainData\\bigcoalThird\\labels\\train\\*.txt")
    # find("D:\\大块煤第二次分组结果\\2006????\\*.txt")
    # moveAll("baiyu")
    # match() 30 72 (133, 364) (133-15, 364-36) (133+15, 364+36)
