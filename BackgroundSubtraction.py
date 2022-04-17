from __future__ import print_function
from operator import truediv
import cv2 as cv
import argparse
import numpy as np
print(cv.__version__)
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
#args = parser.parse_args()
#if args.algo == 'MOG2':
if False:
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

def isLeft(a, b, c):
    return ((b["X"] - a["X"])*(c["Y"] - a["Y"]) - (b["Y"] - a["Y"])*(c["X"] - a["X"])) > 0

width = 1920
height = 1080
#capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
capture = cv.VideoCapture(0)
if not capture.isOpened():
    #print('Unable to open: ' + args.input)
    print('hi')
    exit(0)

i = 0
matrixForGettingIndices = np.zeros((width,height))

w  = capture.get(3)  # float `width`
h = capture.get(4)  # float `height`
print(w,h)
upperTriangleIndices = []
lowerTriangleIndices = []
rightTriangleIndices = []
leftTriangleIndices = []

upX = []
upY = []
lowX = []
lowY = []
rightX = []
rightY = []
leftX = []
leftY = []

for x in range(0,width):
    for y in range(0,height):
        top = True
        bottom = True

        top = isLeft({"X": 0, "Y": 0}, {"X": width, "Y": height}, {"X": x, "Y": y})
       
        bottom = isLeft({"X": width, "Y": 0}, {"X": 0, "Y": height}, {"X": x, "Y": y})
        # print(x, y, top, bottom)
        if top and not bottom:
            # add index to upper triangle indices
            upperTriangleIndices.append((y, x))
            upX.append(x)
            upY.append(y)
        if not top and not bottom:
            # add index to lower triangle indices
            lowerTriangleIndices.append((y, x))
        if top and not bottom:
            # add index to right triangle indices
            rightTriangleIndices.append((y, x))
        if not top and bottom:
            # add index to left triangle indices
            leftTriangleIndices.append((y, x))


# print(upperTriangleIndices)
# upperTriangleIndices = np.triu_indices_from(matrixForGettingIndices)
# upperTriangleIndicesArray = np.asarray(upperTriangleIndices)
# upperTriangleIndicesSet = set((tuple(j) for j in upperTriangleIndicesArray.T.reshape(-1,2)))

# lowerTriangleIndices = np.tril_indices_from(matrixForGettingIndices)
# lowerTriangleIndicesArray = np.asarray(lowerTriangleIndices)
# lowerTriangleIndicesSet = set((tuple(j) for j in lowerTriangleIndicesArray.T.reshape(-1,2)))

# matrixRotatedNinetyClock = np.fliplr(matrixForGettingIndices)

# upperTriRotatedIndices = np.triu_indices_from(matrixRotatedNinetyClock)
# upperTriRotatedIndicesArray = np.asarray(upperTriRotatedIndices)
# upperTriRotatedIndicesSet = set((tuple(j) for j in upperTriRotatedIndicesArray.T.reshape(-1,2)))

# lowerTriRotatedIndices = np.tril_indices_from(matrixRotatedNinetyClock)
# lowerTriRotatedIndicesArray = np.asarray(lowerTriRotatedIndices)
# lowerTriRotatedIndicesSet = set((tuple(j) for j in lowerTriRotatedIndicesArray.T.reshape(-1,2)))

# # #define quadrants
# topIndices = upperTriangleIndicesSet.intersection(upperTriRotatedIndicesSet)
# rightIndices = upperTriangleIndicesSet.intersection(lowerTriRotatedIndicesSet)
# bottomIndices = lowerTriangleIndicesSet.intersection(lowerTriRotatedIndicesSet)
# leftIndices = lowerTriangleIndicesSet.intersection(upperTriRotatedIndicesSet)
[print(j) for j in lowerTriangleIndices]
# print((upX, upY))
while capture.isOpened():
    ret, frame = capture.read()
    if frame is None:
        break
    
    # mirror the frame
    frame = cv.flip(frame, 1)

    # grayscale the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # apply background subtraction

    fgMask = backSub.apply(gray)
    
    #  use upperTriangleIndices to set all values in the upper triangle to 0

    # fgMask[upX, upY] = 255
    
    # cv.rectangle(fgMask, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(fgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
    #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    # draw a red line from the upper-left corner to the lower-right corner
    cv.line(fgMask, (0, 0), (width,height), (255, 0, 0), 3)
    # draw a red line from the upper-right corner to the lower-left corner
    cv.line(fgMask, (width, 0), (0,height), (255, 0, 0), 3)
    # cv.line(fgMask, (10, 2), (100, 20), (255, 255, 255), 3)

    cv.startWindowThread()
    cv.namedWindow('FG Mask')
    #cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    fgMaskMatrix = np.array(fgMask)
    #calculate which quadrants have most difference
    topDiff = [fgMaskMatrix[j] for j in upperTriangleIndices]
    rightDiff = [fgMaskMatrix[j] for j in rightTriangleIndices]
    bottomDiff = [fgMaskMatrix[j] for j in lowerTriangleIndices]
    leftDiff = [fgMaskMatrix[j] for j in leftTriangleIndices]
    
    # print(sum(topDiff))
    # #send that key to pygame
    direction_scores = {sum(topDiff): "up", sum(rightDiff): "right", sum(bottomDiff): "down", sum(leftDiff): "left"}
    max_direction = direction_scores.get(max(direction_scores))
    # print(max_direction)
    # pyautogui.press(max_direction)
    #left and right aren't working, check out their sets or something
    if i % 10 == 0:
        print(max_direction)
        print("top: ", sum(topDiff))
        print("right: ", sum(rightDiff))
        print("bottom: ", sum(bottomDiff))
        print("left: ", sum(leftDiff))
#         #print(upperTriangleIndices)
#         #print(fgMaskMatrix.shape)
# #         print(len(fgMask))
# #         print(len(fgMask[0]))
    #print(fgMask)
    i += 1
    keyboard = cv.waitKey(1)
    if keyboard > 0:
        capture.release()
        break

cv.destroyAllWindows()