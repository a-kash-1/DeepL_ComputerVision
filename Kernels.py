# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:27:37 2018

@author: akash.sharma
"""

from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    # grab the dimensions of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]
    
    # allocate memory for the output iamge, talking care "pad"
    # the borders of the input image so the spatia size (i.e., 
    # width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype = "float")
    
    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            # extract the Region of interest of the image by extracting
            # the center region of the current (x,y)- coordinates
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1] 
            
            # element-wise multiply and sum the values
            k = (roi*K).sum()
            
            # store the convolved value in output
            output[y - pad, x - pad] = k
            
            # rescaling rhe output images in range[0,255] as convolution 
            # makes numbers outside this range
            output = rescale_intensity(output, in_range = (0,255))
            output = (output * 255).astype("uint8")
            
            # return the output range
            return (output)
        
if __name__ == '__main__':

        
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())
    
    smallBlur = np.ones((7,7), dtype = "float") * (1.0/(7*7))
    largeBlur = np.ones((21,21), dtype = "float") * (1.0/(21*21))
    sharpen = np.array(([0,-1,0],[-1,5,-1],[0,-1,0]), dtype = "int")
    laplacian = np.array(([0,1,0],[1,-4,1],[0,1,0]), dtype = "int")
    sobelX = np.array(([-1,0,-1],[-2,0,-2],[-1,0,-1]), dtype = "int")
    sobelY = np.array(([-1,-2,-1],[0,0,0],[1,2,1]), dtype = "int")
    
    
    kernelBank = (("small_blur", smallBlur), ("largeBlur", largeBlur), ("sharpen", sharpen)
     , ("laplacian", laplacian),("sobelX", sobelX),("sobelY", sobelY))
    
    # load the input image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # loop over kernels
    for (kernelName, K) in kernelBank:
        # apply the kernel to the grayscale image using both our custom
        # 'convolve' function and OpenCV's 'filter2D' function
        print("[INFO] applying {} kernel".format(kernelName))
        convolveOutput = convolve(gray, K)
        opencvOutput = cv2.filter2D(gray, -1, K)
        
        # show the output images
        cv2.imshow("Original", gray)
        cv2.imshow("{} - convole".format(kernelName), convolveOutput)
        cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
        
        