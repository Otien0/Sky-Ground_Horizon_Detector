#######################################
######_SKY-GROUND DETECTOR USING_#####
######_OPENCV IN PYTHON_##############
######################################

import cv2
import numpy as np
from scipy import ndimage
from pathlib import Path
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import math
import sys
import glob
import os

#Path to input_images  //may depend on your OS, check on the notes.txt
path = r'/home/net/MORYSO/sky-ground_detection/initial_images/*.*'

#function for calculating pixel data and 3d manipulations for input_image
def get_pixel_dimensions(initial_path, scale= 5.0):
    # loading images from initial_path
    myimage = Image.open([initial_path])
    assert isinstance(myimage, type(None)), 'image not found'
    #Downsampling initial_images
    def block_mean(ar, fact):
        assert isinstance(fact, int), type(fact)
        sx, sy = ar.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy / fact * (X / fact) + Y / fact
        myimage = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
        myimage.shape = (sx / fact, sy / fact)
        return myimage


    #modifying sizes of the new downsampled images
    [xSize, ySize] = myimage.size
    _r = []
    _g = []
    _b = []
    colours = []
    for x in range(0,xSize):
        for y in range(0,ySize):
            #dividing each channel with 255 to change the range of color from 0..255 to 0..1
            [r,g,b] = myimage[x,y]
            r = r/255.0
            _r.append(r)
            g = g/255.0
            _g.append(g)
            b = b/255.0
            _b.append(b)
            colours.append([r,g,b])
    #return colors.rgb([1.0 * x / 255 for x in rgb_tuple])

    #Three-dimensional plotting using Matplotlib
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(_r,_g,_b, c=colours, lw=0)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    fig.add_axes(ax)
    plt.show()
    return

#function calculating the slope-intercept parameterization
def straight_line(m, b, x, y):
    """

    :param m: denotes the slope of the straight_line
    :param b: denotes the y -inter-cept
    :param x: variable describing a specific point
    :param y: variable describing a specific point
    :return:  returns the slope and intercept values
    """
    return y - m*x - b

#setting the range of slope and intercept values
def boundary_detect(img2, xSize, ySize):
    slope = np.linspace(-1,1,50.0)
    inter = np.linspace(0,ySize,50.0)
    
    #initializing variables 
    maximum = []
    J2 = 0

    #iterating through both the slope and y-intercept
    for m in range(len(slope)):
        for b in range(len(inter)):
            #initializing both sky and ground as an array of pixel values
            sk = []
            gn = []

            # iterate over all the pixels in the image and add them to sky and ground
            for i in range(xSize):
                for j in range(ySize):
                    #from optimization criterion technique;
                    #J1 = 1/|Σs| + |Σg|
                    # J2 = 1/|Σs| + |Σg| + ( λ1**s + λ2**s + λ3**s )**2 + ( λ1**g + λ2**g + λ3**g )**2
                    #cross product finding every pixel value above and below the straight_line
                    if((straight_line(slope[m], inter[b], i, j) * (-1 * inter[b])) > 0):
                        sk.append(img2[j, i])
                    else:
                        gn.append(img2[j, i])

            # determining covariance of both sky and ground
            sk = np.transpose(sk)
            gn = np.transpose(gn)
            try:
                cov_s = np.cov(sk)    #calculate covariance of sky
                cov_g = np.cov(gn)    #calculate covariance of ground

                covS = np.linalg.det(cov_s) #calculating determinant of sky
                covG = np.linalg.det(cov_g) #calculating dererminant of ground

                eig_vs, _ = np.linalg.eig(cov_s)  #getting eigenvalues of sky
                eig_vg, _ = np.linalg.eig(cov_g)  #getting eigenvalues of ground

                J = 1/(covS + covG + (eig_vs[0]+eig_vs[1]+eig_vs[2])**2 + (eig_vg[0]+eig_vg[1]+eig_vg[2])**2)

                # get max value of J for all slopes and intercepts
                if J > J2:
                    J2 = J
                    maximum = [slope[m], inter[b]]
                    print(maximum)
            except Exception:
                pass

    return maximum

# Displays the output_images on new window
def display_output(Image, image):
    cv2.namedWindow(Image)
    cv2.imshow(Image, image)

# Plots the straight_line coming out of a Hough Line Transform
def get_line(img2, horizon, path):
    #getting width of initial input_image
    xSize = img2.shape[1]
    print("xSize", xSize)
    m = horizon[0]
    b = horizon[1]
    #modify the slope
    y2 = int(m*(xSize-1)+b)
    #drawing line across the initial input_image at point of horizon;
    #this will be a blue line, with thickness of the line = 4
    cv2.line(img2, (0, int(b)), (xSize - 1, y2), (255, 10, 10), 4)
    display_output("output_image", img2)
    cv2.waitKey(5)


#Setting the current working directory
dir_name = Path.cwd()
dir_name = path

initial_path = []
#setting initial_path as list in cwd
initial_path = [dir_name]

#adding CWD to initial_path as a tupple
initial_path.append(dir_name)

#looping through the initial_path
for path in initial_path:
    path = r'/home/net/MORYSO/sky-ground_detection/initial_images/*.*'
    #try:

    #scale factor as float value
    scale = 5.0
    img = 1
    for img, file in enumerate(glob.glob(path)):
        print(img, file)

        #reading image from initial_images path
        a = cv2.imread(file)
        img2_original = cv2.imread(file)
        # except(RuntimeError, TypeError, NameError):
        #Initial Dimensions of input_image
        ySize = img2_original.shape[0]
        xSize = img2_original.shape[1]
        #Resizing image by changing its dimension
        # where:
        # fx = scale factor along the horizontal axis
        # fy = scale factor along the vertical axis
        img2 = cv2.resize(img2_original, (0, 0), fx=1 / scale, fy=1 / scale)
        #Dimensions of image after Downsampling/resizing
        ySize = img2.shape[0]
        xSize = img2.shape[1]

        #Detecting the horizon/sky segmentation
        horizon = []
        horizon = boundary_detect(img2, xSize, ySize)

        #get scale factor and line inputs to draw the required blue line across
        horizon[1] *= scale
        get_line(img2_original, horizon, file)

        #Display the output_image
        c = img2_original
        cv2.imshow('output_image', c)
        # Save image to output directory/path
        cv2.imwrite('/home/net/MORYSO/sky-ground_detection/output_images/output_image{}.jpg'.format(img), c)
        img += 1

        #wait for 1000m/s after display before closing window
        cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print('sky-ground-detector')





