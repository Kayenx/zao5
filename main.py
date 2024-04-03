#!/usr/bin/python

import sys
from tkinter import font
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

sobel_x = np.array([
    [ -1, 0, 1],
    [ -2, 0, 2],
    [ -1, 0, 1] ])
    
sobel_y = np.array([
    [ -1, -2, -1],
    [  0,  0,  0],
    [  1,  2,  1] ])
    
def main(argv):

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
   
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)
    
 
    test_images = [img for img in glob.glob("test_images_zao/*.jpg")]
    test_images.sort()
    print("********************************************************")
  
    recognizer = cv2.face.LBPHFaceRecognizer.create(2,12,8,8)

    free_spot_images = []
    for img_path in glob.glob("free/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        free_spot_images.append((img, 1))  

    full_spot_images = []
    for img_path in glob.glob("full/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        full_spot_images.append((img, 0))  
        
    images = np.array([data[0] for data in free_spot_images] + [data[0] for data in full_spot_images])
    labels = np.array([data[1] for data in free_spot_images] + [data[1] for data in full_spot_images])
    recognizer.train(images, labels)
    cv2.namedWindow("image",0)

    tp_canny = 0
    fp_canny = 0
    fn_canny = 0
        

    for image_name in test_images:
        image = cv2.imread(image_name,1)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        with open(image_name[0:-4] + ".txt", "r") as f:
            y_true = [int(line) for line in f]
        
        i = 0
        for cord in pkm_coordinates:
            one_place = four_point_transform(image_gray,cord)
            one_place = cv2.resize(one_place,(86,86))

            #cv2.imwrite("template.png",one_place)
            #result = cv2.matchTemplate(one_place, temp_mat, cv2.TM_CCORR_NORMED)
            #min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)
            label,confidence = recognizer.predict(one_place)
            if(label ==1 and confidence >= 99):
                cv2.drawContours(image,[np.array(cord).reshape((-1,1,2)).astype(int)],-1,(0,255,0),3)
                if(y_true[i]==0):
                    tp_canny = tp_canny + 1
                else:
                    fp_canny = fp_canny + 1

            else:
                cv2.drawContours(image,[np.array(cord).reshape((-1,1,2)).astype(int)],-1,(0,0,255),3)
                if(y_true[i]==0):
                    fn_canny = fn_canny + 1            
            
            
            '''
            if (tp + fp != 0 and tp + fn != 0):
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision* recall) / (precision + recall)
                cv2.putText(image, f"F1 Score: {f1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            '''
            cv2.imshow("image",image)
            i= i + 1 
        cv2.waitKey(0)    
        
    precision_total = tp_canny / (tp_canny + fp_canny)
    recall_total = tp_canny / (tp_canny + fn_canny)
    f1_total = 2 * (precision_total* recall_total) / (precision_total + recall_total)  
    print("\n Total f1 : ")  
    print(f1_total)
    


if __name__ == "__main__":
   main(sys.argv[1:])     
