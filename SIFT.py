import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import filecmp

df = pd.DataFrame(columns = {'filename1','filename2','keypts1','keypts2','match_points','matchpercent'})


#reading image

def orb_matching(file1,file2):
    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)

    img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_bw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # keypoints
    orb = cv2.ORB_create()
    keypoints_1, descriptors_1 = orb.detectAndCompute(img1_bw, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2_bw, None)

    percent = 0

    # feature matching
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = matcher.match(descriptors_1, descriptors_2)
    #matches = sorted(matches, key=lambda x: x.distance)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            a = len(good)
            percent = (a * 100) / len(keypoints_2)

    x1 = len(keypoints_1)
    x2 = len(keypoints_2)
    x3 = len(matches)

    return [file1,file2,x1,x2,x3,percent]


# assign directory
directory = 'data1'

# iterate over files in
# that directory
for filename1 in os.scandir(directory):
    for filename2 in os.scandir(directory):
        if not(filecmp.cmp(filename1.path,filename2.path)):
            f1 = filename1.path
            f2 = filename2.path

            list1 = orb_matching(f1, f2)
            df.loc[len(df)] = list1
            print(list1)


print(df)

df.to_csv(r'E:\..\..\sift11.csv', index=False)