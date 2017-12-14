from __future__ import division
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# function to calculate gaussian using the formula
def gaussian(sigma,x):
    a= 1/(np.sqrt(2*np.pi)*sigma)
    b=math.exp(-(x**2)/(2*(sigma**2)))
    c = a*b
    return a*b

# getting kernel using [-1,0,1] and sigma
def gaussian_kernel(sigma):
    a=gaussian(sigma, -1)
    b=gaussian(sigma, 0)
    c=gaussian(sigma, 1)
    sum=a+b+c
    if sum!=0:
        a=a/sum
        b=b/sum
        c=c/sum
    return np.reshape(np.asarray([a,b,c]), (1,3))

# function to calculate 1st order derivative of gaussian
def gaussian_derivative(sigma,x):
    a = -x/ (np.sqrt(2 * np.pi)*sigma**3)
    b = math.exp(-(x ** 2) / (2 * (sigma ** 2)))
    c = a * b
    return a * b

## getting kernel from 1st order derivative of gaussian for [-1,0,1]
def gaussian_derivative_kernel(sigma):
    a=gaussian_derivative(sigma, -1)
    b=gaussian_derivative(sigma, 0)
    c=gaussian_derivative(sigma, 1)
    sum=a+b+c
    if sum!=0:
        a=a/sum
        b=b/sum
        c=c/sum
    return np.reshape(np.asarray([a,b,c]), (1,3))


# To convolve image with kernel
# def convolve_xdirection(image,kernel):
#     Ix = image.copy()
#     for i in range(0, Ix.shape[0]):
#         for j in range(0, Ix.shape[1] - 1):
#             Ix[i, j] = (Ix[i, j - 1] * kernel[0] + Ix[i, j] * kernel[1] + Ix[i, j + 1] * kernel[2])/np.sum(kernel)
#     return Ix
# def convolve_ydirection(image,kernel):
#     Iy=image.copy()
#     shape = image.shape
#
#     for y in range(0, shape[1]):
#         for x in range(0, shape[0] - 1):
#             Iy[x, y] = (Iy[x - 1, y] * kernel[0] + Iy[x, y] * kernel[1] + Iy[x + 1, y] * kernel[2])/np.sum(kernel)
#
#     return Iy

# function to calculate magnitude of image
def magnitude_nms(imagex,imagey):
    Mag = np.zeros(imagex.shape)

    for i in range(imagex.shape[0]):
        for j in range(imagex.shape[1]):
            Mag[i,j]=np.sqrt((((imagex[i][j] ** 2) + ((imagey[i][j]) ** 2))))

    return Mag

#function to calculate direction in degrees
def orientation_nms(imagex,imagey):
    Ori = imagex.copy()
    for i in range(imagex.shape[0]):
        for j in range(imagex.shape[1]):
            Ori[i, j] = (180/np.pi)*np.arctan2(imagey[i,j], imagex[i,j])  # converting it into degrees also
    return Ori

# non maximal suppression.
""""
 1. if theta is between 0 to 22.5 or between 157.5 to 202.5 or between 337.5 to 360 than taking (1,0) and (-1,0) as values in gradient direction
 2. if theta is between 22.5 to 67.5 or between 202.5 to 247.5  than taking (1,1) and (-1,-1) as values in gradient direction
 3. if theta is between 67.5 to 112.5 or between 247.5 to 292.5  than taking (0,1) and (0,-1) as values in gradient direction
 4.if theta is between 112.5 to 157.5 or between 292.5 to 337.5 than taking (-1,1) and (1,-1) as values in gradient direction
 

"""
def nms(magnitude, orient):
    nms = np.zeros(magnitude.shape, magnitude.dtype)

    for x in range(1,magnitude.shape[0]-1):
        for y in range(1,magnitude.shape[1]-1):
            theta = orient[x,y]
            if (0.0 <= theta <= 22.5) or (157.5 < theta <= 202.5) or (337.5 < theta <= 360):
                 if magnitude[x, y] > magnitude[x + 1, y ] and magnitude[x, y] > magnitude[x - 1, y]:
                     nms[x, y] = magnitude[x, y]
            elif (22.5 < theta <= 67.5) or (202.5 < theta<= 247.5):
                if magnitude[x, y] > magnitude[x + 1, y +1] and magnitude[x, y] > magnitude[x - 1, y - 1]:
                    nms[x, y] = magnitude[x, y]
            elif (67.5 < theta <= 112.5) or (247.5 < theta <= 292.5):
                if magnitude[x, y] > magnitude[x, y + 1] and magnitude[x, y] > magnitude[x , y - 1]:
                    nms[x, y] = magnitude[x, y]
            elif (112.5 < theta <= 157.5) or (292.5 < theta <= 337.5):
                if magnitude[x, y] > magnitude[x -1, y + 1] and magnitude[x, y] > magnitude[x + 1, y - 1]:
                    nms[x, y] = magnitude[x, y]
            else:
                break
    return nms


# hysteris thresholding
"""
if intensity is in less than low than marking it to zero, else if intensity is greater than high than its a edge pixel, marking it as 255. 
For intensity between low and high, if any neighbour is connected to above high than marking it as edge pizel
"""
def hysteresis_threshold(nms, low, high):
    hys = nms.copy()
    for i in range(hys.shape[0]):
        for j in range(hys.shape[1]):
            if hys[i,j]<=low:
                hys[i,j]=0
            elif hys[i,j]>=high:
                hys[i,j]=255
            elif hys[i,j]>low and hys[i,j]<high:
                if i > 0 and i <hys.shape[0]-1 and j > 0 and j <hys.shape[1]-1:
                    if ((hys[i + 1][j + 1] >= high) or (hys[i - 1][j - 1] >= high) or (hys[i][j - 1] >= high) or (
                                hys[i - 1][j] >= high) or (hys[i + 1][j - 1] >= high)  or (hys[i - 1][j + 1] >= high) or (hys[i + 1][j] >= high) or (hys[i][j + 1] >= high)  ):
                        hys[i][j] = 255
    return hys

# Plotting all images
def plot_image(img1,title1,img2,title2,img3,title3,img4,title4,img5,title5,img6,title6,img7,title7,img8,title8):

    plt.subplot(331)
    plt.imshow(img1,cmap=cm.gray)
    plt.title(title1)


    plt.subplot(332)
    plt.imshow(img2, cmap=cm.gray)
    plt.title(title2)


    plt.subplot(333)
    plt.imshow(img3, cmap=cm.gray)
    plt.title(title3)


    plt.subplot(334)
    plt.imshow(img4, cmap=cm.gray)
    plt.title(title4)

    plt.subplot(335)
    plt.imshow(img5, cmap=cm.gray)
    plt.title(title5)


    plt.subplot(336)
    plt.imshow(img6, cmap=cm.gray)
    plt.title(title6)


    plt.subplot(337)
    plt.imshow(img7, cmap=cm.gray)
    plt.title(title7)


    plt.subplot(338)
    plt.imshow(img8, cmap=cm.gray)
    plt.title(title8)
    plt.show()

# main function
def canny_edge(filepath,sigma):
    I=cv.imread(filepath,0)

    G=gaussian_kernel(sigma)    # one dimensional gaussian mask to convolve with I [-1,0,1]
    Gx=gaussian_derivative_kernel(sigma) # calculating gaussian 1d derivative with sigma as 1 and x =-1,0,1
    Gy=np.transpose(Gx)
    Ix=cv.filter2D(I,-1,G)
    Iy=cv.filter2D(I,-1,np.transpose(G))
    Ixx=cv.filter2D(Ix,-1,Gx)

    # print Ixx
    Iyy=cv.filter2D(Iy,-1,Gy)

    magnitude= magnitude_nms(Ixx,Iyy) # magnitude of the edge response by combining x and y component


    theta = orientation_nms(Ixx,Iyy) # direction

    nms_result=nms(magnitude,theta)
   # nms_result *= 255.0 / nms_result.max()
    final_result =hysteresis_threshold(nms_result,5,10)

    plot_image(Ix, "Smoothing in x direction",Iy, "Smoothing in y direction",Ixx, "Convolution of Ix with Gx",Iyy, "Convolution of Iy with Gy",theta,"Orientation",magnitude,"Magnitude",nms_result,"Non Maximal Suppression",final_result,"Final Result After Hysteresis Thresholding")

"""
 
1.	Taken output with 3 images and sigma=0.5, 0.75, 1.0
2.	What we can see is if sigma is low than we get little bit noisy images and where is if we increase sigma we start losing edges.
      
      For me, sigma=0.75 worked best as per visual interpretation 
"""
canny_edge("question1_1.jpg",0.5)
canny_edge("question1_1.jpg",0.75)
canny_edge("question1_1.jpg",1)
canny_edge("question1_3.jpg",0.5)
canny_edge("question1_3.jpg",0.75)
canny_edge("question1_3.jpg",1)
canny_edge("question1_2.jpg",0.5)
canny_edge("question1_2.jpg",0.75)
canny_edge("question1_2.jpg",1)

