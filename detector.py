#######################################
######_SKY-GROUND DETECTOR USING_#####
######_OPENCV IN PYTHON_##############
######################################

import cv2
import numpy as np
from pathlib import Path
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import sys
import glob
import os

path = r'/home/net/MORYSO/sky-ground_detection/initial_images/*.*'

def plotPixelData(initial_path, scale):
    myimage = Image.open([initial_path])
    assert isinstance(myimage, type(None)), 'image not found'
    [xSize, ySize] = myimage.size
    myimage = myimage.resize((int(xSize/scale),int(ySize/scale)), PIL.Image.LANCZOS)
    myimage = myimage.load()
    return myimage

    [xSize, ySize] = myimage.size
    _r = []
    _g = []
    _b = []
    colours = []
    for x in range(0,xSize):
        for y in range(0,ySize):
            [r,g,b] = myimage[x,y]
            r /= 255.0
            _r.append(r)
            g /= 255.0
            _g.append(g)
            b /= 255.0
            _b.append(b)
            colours.append([r,g,b])

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(_r,_g,_b, c=colours, lw=0)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    fig.add_axes(ax)
    plt.show()
    return

def line(m, b, x, y):
    return y - m*x - b

def detectHorizon(cvImg, xSize, ySize):
    res = 100.0
    slope = np.linspace(-1,1,res)
    inter = np.linspace(0,ySize,res)
    maximum = []
    J_max = 0

    for m in range(len(slope)):
        for b in range(len(inter)):
            sky = []
            gnd = []

            for i in range(xSize):
                for j in range(ySize):
                    if((line(slope[m],inter[b],i,j)*(-1*inter[b])) > 0):
                        sky.append(cvImg[j,i])
                    else:
                        gnd.append(cvImg[j,i])


            sky = np.transpose(sky)
            gnd = np.transpose(gnd)
            try:
                co_s = np.cov(sky)
                co_g = np.cov(gnd)
                co_sD = np.linalg.det(co_s)
                co_gD = np.linalg.det(co_g)
                eig_s, _ = np.linalg.eig(co_s)
                eig_g, _ = np.linalg.eig(co_g)

                J = 1/(co_sD + co_gD + (eig_s[0]+eig_s[1]+eig_s[2])**2 + (eig_g[0]+eig_g[1]+eig_g[2])**2)
                if J > J_max:
                    J_max = J
                    maximum = [slope[m], inter[b]]
                    print(maximum)
            except Exception:
                pass

    return maximum

def display_image(Image, image):
    cv2.namedWindow(Image)
    cv2.imshow(Image, image);


def plot_line(cvImg, horizon, path):
    xSize = cvImg.shape[1]
    print("xSize", xSize)
    m = horizon[0]
    b = horizon[1]
    y2 = int(m*(xSize-1)+b)
    cv2.line(cvImg, (0,int(b)), (xSize-1, y2), (10,10,255), 5)
    display_image("output_image", cvImg)


dir_name = sys.argv[0]
dir_name = Path.cwd()
dir_name = "/home/net/MORYSO/sky-ground_detection/initial_images"
#print(dir_name)

initial_path = []
initial_path = [dir_name]

initial_path.append(dir_name)
scale = 10.0

for path in initial_path:
    path = r'/home/net/MORYSO/sky-ground_detection/initial_images/*.*'
    print("Accessing images in initial_path")
    print("Processing images: ", path)
    #try:
    img = 1
    for img, file in enumerate(glob.glob(path)):
        print(img, file)
        a = cv2.imread(file)
        cvImg_original = cv2.imread(file)
        # except(RuntimeError, TypeError, NameError):
        ySize = cvImg_original.shape[0]
        xSize = cvImg_original.shape[1]
        cvImg = cv2.resize(cvImg_original, (0, 0), fx=1 / scale, fy=1 / scale)
        ySize = cvImg.shape[0]
        xSize = cvImg.shape[1]
        cvImg = cv2.GaussianBlur(cvImg, (5, 5), 0)

        horizon = []
        horizon = detectHorizon(cvImg, xSize, ySize)

        horizon[1] *= scale
        plot_line(cvImg_original, horizon, file)

        c = cvImg_original
        cv2.imshow('output_image', c)
        cv2.imwrite('/home/net/MORYSO/sky-ground_detection/output_images/output_image{}.jpg'.format(img), c)
        img += 1

        cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print('sky-ground-detector')





