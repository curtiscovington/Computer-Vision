from __future__ import print_function
from operator import truediv
import cv2 as cv
import argparse
import numpy as np
print(cv.__version__)


# detects if point c is to the left (counter-clockwise) of the line segment formed by points a and b
def isLeft(a, b, c):
    return ((b["X"] - a["X"])*(c["Y"] - a["Y"]) - (b["Y"] - a["Y"])*(c["X"] - a["X"])) > 0


class ColorDetector():
    colorMode = True
    width = 1920
    height = 1080
    capture = None
    top = []
    bot = []
    left = []
    right = []

    # some blue
    colorLower = np.array([110,235,165], np.uint8)
    colorUpper = np.array([130,255,255], np.uint8)

    threshold = 10

    selectLower = True
    selectedColor = None
    hRange = 10
    sRange = 10
    vRange = 40

    def __init__(self):
        for x in range(0,self.width):
            for y in range(0,self.height):
                leftLine = True
                rightLine = True

                leftLine = isLeft({"X": 0, "Y": 0}, {"X": self.width, "Y": self.height}, {"X": x, "Y": y})
            
                rightLine = isLeft({"X": self.width, "Y": 0}, {"X": 0, "Y": self.height}, {"X": x, "Y": y})
                # print(x, y, top, bottom)
                if leftLine and rightLine:
                    # add index to upper triangle indices
                    self.left.append((y, x))
                elif not leftLine and not rightLine:
                    # add index to lower triangle indices
                    self.right.append((y, x))
                elif leftLine and not rightLine:
                    # add index to right triangle indices
                    self.bot.append((y, x))
                elif not leftLine and rightLine:
                    # add index to left triangle indices
                    self.top.append((y, x))
    
    def onClick(self, event, x, y, flags, param):
        
        if event == cv.EVENT_LBUTTONDOWN:
            ret, frame = self.capture.read()
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            pixel = hsv[y, x]
            lH = min(max(0, pixel[0] - self.hRange), 179)
            lS = min(max(0, pixel[1] - self.sRange), 255)
            lV = min(max(0, pixel[2] - self.vRange), 255)
            uH = min(max(0, pixel[0] + self.hRange), 179)
            uS = min(max(0, pixel[1] + self.sRange), 255)
            uV = min(max(0, pixel[2] + self.vRange), 255)
            self.colorLower = np.array([lH, lS, lV])
            self.colorUpper = np.array([uH, uS, uV])

            # convert pixel to rgb
            
            print(f"{frame[y, x][2]}, {frame[y, x][1]}, {frame[y, x][0]}")
        # if event == cv.EVENT_LBUTTONUP:
        #     ret, frame = self.capture.read()
        #     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #     self.end = (x, y)
            
        #     # get all the pixels in the rectangle defined by the start and end points
        #     # and convert them to a numpy array
        #     pixels = np.array(list(map(lambda x: hsv[x[1], x[0]], self.getRectangle(self.start, self.end))))

        #     # find the min and max values of the pixels
        #     minVal = np.min(pixels, axis=0)
        #     maxVal = np.max(pixels, axis=0)

        #     # find the average of the pixels
        #     avg = np.average(pixels, axis=0)
            
        #     self.colorLower = np.array([avg[0] - self.threshold, avg[1], avg[2]], np.uint8)
        #     self.colorUpper = np.array([avg[0] + self.threshold, avg[1], avg[2]], np.uint8)
            
        #     self.selectedColor = avg

        #     print(hsv[y, x])
        #     print(self.selectedColor, minVal)
            # 

            # pixel = hsv[y, x]
            
            # self.colorLower = np.array([pixel[0] - self.threshold, pixel[1], pixel[2]], np.uint8)
            # self.colorUpper = np.array([pixel[0] + self.threshold, pixel[1], pixel[2]], np.uint8)

    def getRectangle(self, start, end):
        # get the top left and bottom right points
        x1 = min(start[0], end[0])
        x2 = max(start[0], end[0])
        y1 = min(start[1], end[1])
        y2 = max(start[1], end[1])
        # return the points
        return [(x, y) for x in range(x1, x2) for y in range(y1, y2)]

    def onHTrackbarChange(self, val):
        self.hRange = val
    def onSTrackbarChange(self, val):
        self.sRange = val
    def onVTrackbarChange(self, val):
        self.vRange = val
   
    def run(self):


        self.capture = cv.VideoCapture(0)
        if not self.capture.isOpened():
            exit(0)

        cv.startWindowThread()
        cv.namedWindow('window')
        cv.setMouseCallback("window", self.onClick)
        cv.namedWindow('detector')

        cv.namedWindow('HSV Range')
        cv.createTrackbar("H", "HSV Range", self.hRange, 179, self.onHTrackbarChange)
        cv.createTrackbar("S", "HSV Range", self.sRange, 255, self.onSTrackbarChange)
        cv.createTrackbar("V", "HSV Range", self.vRange, 255, self.onVTrackbarChange)
        
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if frame is None:
                break
            # mirror the frame
            frame = cv.flip(frame, 1)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.colorLower, self.colorUpper)

            
            # cv.line(fgMask, (10, 2), (100, 20), (255, 255, 255), 3)
            # display text on the frame
            # if self.selectedColor:
            #     cv.putText(frame, f'Selected Color: {self.selectedColor[0]} {self.selectedColor[1]} {self.selectedColor[2]}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv.putText(frame, "Select Lower Color", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # cv.putText(frame, f'Threshold value is set to {self.threshold}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # draw a red line from the upper-left corner to the lower-right corner
            cv.line(mask, (0, 0), (self.width,self.height), (255, 0, 0), 3)
            # draw a red line from the upper-right corner to the lower-left corner
            cv.line(mask, (self.width, 0), (0,self.height), (255, 0, 0), 3)

            #cv.imshow('Frame', frame)
            cv.imshow('window', frame)
            
            cv.imshow('detector', mask)

            fgMaskMatrix = np.array(mask)
            #calculate which quadrants have most difference
            topDiff = [fgMaskMatrix[j] for j in self.top]
            rightDiff = [fgMaskMatrix[j] for j in self.right]
            bottomDiff = [fgMaskMatrix[j] for j in self.bot]
            leftDiff = [fgMaskMatrix[j] for j in self.left]
            
            # #send that key to pygame
            direction_scores = {sum(topDiff): "up", sum(rightDiff): "right", sum(bottomDiff): "down", sum(leftDiff): "left"}
            max_direction = direction_scores.get(max(direction_scores))
            print(max_direction)

            keyboard = cv.waitKey(1)
            if keyboard > 0:
                # if the + is pressed
                if keyboard == ord('+'):
                    self.threshold += 1
                # if the - is pressed
                elif keyboard == ord('-'):
                    self.threshold -= 1
                # if the 'q' key is pressed
                elif keyboard == ord('q'):
                    self.capture.release()
                    break



# while capture.isOpened():
    
    
colorDetector = ColorDetector()
colorDetector.run()


#     # set the frame value to 255 if red is greater than threshold
#     frame[:,:,2] = np.where(frame[:,:,0] > bThreshold, 255, 0)
#     frame[:,:,1] = np.where(frame[:,:,0] > bThreshold, 255, 0)
#     frame[:,:,0] = np.where(frame[:,:,0] > bThreshold, 255, 0)
# print(frame[(0,0)])

# convert any red above the threshold to white, everything else to black
# frame[:,:,0] = np.where(frame[:,:,0] > threshold, 255, 0)
# frame[:,:,1] = np.where(frame[:,:,1] > threshold, 255, 0)
# frame[:,:,2] = np.where(frame[:,:,2] > threshold, 255, 0)

#     # grayscale the frame
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # apply background subtraction

#     fgMask = gray#backSub.apply(gray)
    
#     # Use this to check if the triangles are correct 
#     # set all values in top to 255
#     # split too to rows cols array
#     # fgMask[(np.array(bot)[:,0], np.array(bot)[:, 1])] = 255
#     # fgMask[upX, upY] = 255
    
#     # cv.rectangle(fgMask, (10, 2), (100,20), (255,255,255), -1)
#     # cv.putText(fgMask, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#     #            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

#     # draw a red line from the upper-left corner to the lower-right corner
#     cv.line(fgMask, (0, 0), (width,height), (255, 0, 0), 3)
#     # draw a red line from the upper-right corner to the lower-left corner
#     cv.line(fgMask, (width, 0), (0,height), (255, 0, 0), 3)
#     # cv.line(fgMask, (10, 2), (100, 20), (255, 255, 255), 3)

    
    
    
    # fgMaskMatrix = np.array(fgMask)
    # #calculate which quadrants have most difference
    # topDiff = [fgMaskMatrix[j] for j in top]
    # rightDiff = [fgMaskMatrix[j] for j in right]
    # bottomDiff = [fgMaskMatrix[j] for j in bot]
    # leftDiff = [fgMaskMatrix[j] for j in left]
    
    # # #send that key to pygame
    # direction_scores = {sum(topDiff): "up", sum(rightDiff): "right", sum(bottomDiff): "down", sum(leftDiff): "left"}
    # max_direction = direction_scores.get(max(direction_scores))
    # print(max_direction)
#     # pyautogui.press(max_direction)
#     #left and right aren't working, check out their sets or something
#     # if i % 10 == 0:
#     #     print(max_direction)
#     #     print("top: ", sum(topDiff))
#     #     print("right: ", sum(rightDiff))
#     #     print("bottom: ", sum(bottomDiff))
#     #     print("left: ", sum(leftDiff))
# #         #print(upperTriangleIndices)
# #         #print(fgMaskMatrix.shape)
# # #         print(len(fgMask))
# # #         print(len(fgMask[0]))
#     #print(fgMask)
#     i += 1
    

cv.destroyAllWindows()