from __future__ import print_function
from operator import truediv
import cv2 as cv
import argparse
import numpy as np
import pyautogui
print(cv.__version__)


# detects if point c is to the left (counter-clockwise) of the line segment formed by points a and b
def isLeft(a, b, c):
    return ((b["X"] - a["X"])*(c["Y"] - a["Y"]) - (b["Y"] - a["Y"])*(c["X"] - a["X"])) > 0

backSub = cv.createBackgroundSubtractorKNN()
class Detector():
    colorMode = True
    capture = None
    top = []
    bot = []
    left = []
    right = []

    defaultLower =  np.array([15,141,172], np.uint8)
    defaultUpper =  np.array([26,203,248], np.uint8)
    
    # some blue
    colorLower = np.array(defaultLower, copy=True)
    colorUpper = np.array(defaultUpper, copy=True)
    color = None

    threshold = 10

    selectLower = True
    selectedColor = None
    hRange = 10
    sRange = 10
    vRange = 40
    
    lastClick = None
    testImg = None
    mode = 0
    current_direction = None
    
    useBackgroundSubtraction = False
    def __init__(self):
        self.hRange = 0
        self.sRange = 0
        self.vRange = 0

        # create an artificial image where half the image is colorLower and half is colorUpper
        # this is to test the color thresholding
        self.testImg = np.zeros((400, 400, 3), np.uint8)
        self.testImg[:, :, 0] = np.arange(0, 400, dtype=np.uint8)
        self.testImg[:, :, 1] = np.arange(0, 400, dtype=np.uint8)
        self.testImg[:, :, 2] = np.arange(0, 400, dtype=np.uint8)

        self.capture = cv.VideoCapture(0)
        if not self.capture.isOpened():
            exit(0)

        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))

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
    
    def updateTestImage(self):
        
        # convert colorLower pixel and colorUpper pixel from hsv to bgr
        
        # colorLower = cv.cvtColor(np.uint8([[self.colorLower]]), cv.COLOR_HSV2BGR)
        # colorUpper = cv.cvtColor(np.uint8([[self.colorUpper]]), cv.COLOR_HSV2BGR)
        # everything below y = 100 is the colorLower
        # everything above y = 100 is the colorUpper
        self.testImg[:, :] = self.colorUpper
        self.testImg[(200,200)[0]:, :] = self.colorLower

        # put the hsv value on the image
        cv.putText(self.testImg, str(self.colorUpper), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(self.testImg, str(self.colorLower), (10, 230), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.testImg = cv.cvtColor(self.testImg, cv.COLOR_HSV2BGR)
        
        cv.setTrackbarPos("-H", "HSV Range", self.colorLower[0])
        cv.setTrackbarPos("-S", "HSV Range", self.colorLower[1])
        cv.setTrackbarPos("-V", "HSV Range", self.colorLower[2])
        cv.setTrackbarPos("+H", "HSV Range", self.colorUpper[0])
        cv.setTrackbarPos("+S", "HSV Range", self.colorUpper[1])
        cv.setTrackbarPos("+V", "HSV Range", self.colorUpper[2])

    def onClick(self, event, x, y, flags, param):
        
        if event == cv.EVENT_LBUTTONDOWN:
            self.lastClick = (x, y)
            _, frame = self.capture.read()

            # frame[y, x] = [255, 0, 0]
            # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            # get pixels in a 3 x 3 grid around the click
           
            
            pixel = hsv[y, x]
            lH = min(max(0, pixel[0] - self.hRange), 179)
            lS = min(max(0, pixel[1] - self.sRange), 255)
            lV = min(max(0, pixel[2] - self.vRange), 255)
            uH = min(max(0, pixel[0] + self.hRange), 179)
            uS = min(max(0, pixel[1] + self.sRange), 255)
            uV = min(max(0, pixel[2] + self.vRange), 255)
            # set trackbars to the values of the clicked pixel
            cv.setTrackbarPos("-H", "HSV Range", lH)
            cv.setTrackbarPos("+H", "HSV Range", uH)
            cv.setTrackbarPos("-S", "HSV Range", lS)
            cv.setTrackbarPos("+S", "HSV Range", lS)
            cv.setTrackbarPos("-V", "HSV Range", lV)
            cv.setTrackbarPos("+V", "HSV Range", lV)
            self.colorLower = np.array([lH, lS, lV])
            self.colorUpper = np.array([uH, uS, uV])

    def getRectangle(self, start, end):
        # get the rectangle coordinates from the start and end points
        # get the top left and bottom right points
        x1 = min(start[0], end[0])
        x2 = max(start[0], end[0])
        y1 = min(start[1], end[1])
        y2 = max(start[1], end[1])

        # get the pixels in the rectangle
        points = []
        for x in range(x1, x2):
            for y in range(y1, y2):
                points.append((y, x))
        return np.array(points)

    def onLHTrackbarChange(self, val):
        self.colorLower[0] = val
    def onUHTrackbarChange(self, val):
        self.colorUpper[0] = val
    def onLSTrackbarChange(self, val):
        self.colorLower[1] = val
    def onUSTrackbarChange(self, val):
        self.colorUpper[1] = val
    def onLVTrackbarChange(self, val):
        self.colorLower[2] = val
    def onUVTrackbarChange(self, val):
        self.colorUpper[2] = val
   
    def run(self):
        cv.startWindowThread()
        cv.namedWindow('detector')
        cv.setMouseCallback("detector", self.onClick)

        cv.namedWindow('HSV Range')
        self.updateTestImage()
        cv.createTrackbar(f"-H", "HSV Range", self.hRange, 179, self.onLHTrackbarChange)
        cv.createTrackbar(f"+H", "HSV Range", self.hRange, 179, self.onUHTrackbarChange)
        cv.createTrackbar(f"-S", "HSV Range", self.sRange, 255, self.onLSTrackbarChange)
        cv.createTrackbar(f"+S", "HSV Range", self.sRange, 255, self.onUSTrackbarChange)
        cv.createTrackbar(f"-V", "HSV Range", self.vRange, 255, self.onLVTrackbarChange)
        cv.createTrackbar(f"+V", "HSV Range", self.vRange, 255, self.onUVTrackbarChange)

        i = 0
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if frame is None:
                break
            self.updateTestImage()
            # mirror the frame
            frame = cv.flip(frame, 1)
            if self.useBackgroundSubtraction:
                mask = backSub.apply(frame)
            else:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, self.colorLower, self.colorUpper)

            if self.lastClick is not None:
                cv.circle(frame, self.lastClick, 5, (0, 0, 255), -1)

            if self.current_direction is not None:
                cv.putText(frame, self.current_direction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # combine the mask with the frame
            

            if self.mode == 0:
                cv.imshow('detector', frame)
            elif self.mode == 1:
                # draw a red line from the upper-left corner to the lower-right corner
                cv.line(mask, (0, 0), (self.width,self.height), (255, 0, 0), 3)
                # draw a red line from the upper-right corner to the lower-left corner
                cv.line(mask, (self.width, 0), (0,self.height), (255, 0, 0), 3)
                cv.imshow('detector', mask)
            else:
                res = cv.bitwise_and(frame, frame, mask=mask)
                if self.current_direction is not None:
                    cv.putText(res, self.current_direction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('detector', res)

            cv.imshow('HSV Range', self.testImg)

            if i % 15 == 0:
                fgMaskMatrix = np.array(mask)
                #calculate which quadrants have most difference
                topDiff = [fgMaskMatrix[j] for j in self.top]
                rightDiff = [fgMaskMatrix[j] for j in self.right]
                bottomDiff = [fgMaskMatrix[j] for j in self.bot]
                leftDiff = [fgMaskMatrix[j] for j in self.left]
                
                # #send that key to pygame
                direction_scores = {sum(topDiff): "up", sum(rightDiff): "right", sum(bottomDiff): "down", sum(leftDiff): "left"}
                max_direction = direction_scores.get(max(direction_scores))
                self.current_direction = max_direction
                # print(max_direction)
                pyautogui.press(max_direction)
            i = i + 1
            keyboard = cv.waitKey(1)
            if keyboard > 0:
                # if the + is pressed
                if keyboard == ord('+'):
                    self.threshold += 1
                # if the - is pressed
                elif keyboard == ord('-'):
                    self.threshold -= 1
                elif keyboard == ord('r'):
                    self.colorLower = np.array(self.defaultLower, copy=True)
                    self.colorUpper = np.array(self.defaultUpper, copy=True)
                    
                    self.updateTestImage()
                elif keyboard == ord('m'):
                    # change the mode
                    self.mode = (self.mode + 1) % 3
                # space is pressed
                elif keyboard == ord(' '):
                    pyautogui.press("space")
                elif keyboard == ord('b'):
                    self.useBackgroundSubtraction = not self.useBackgroundSubtraction
                # if the 'q' key is pressed
                elif keyboard == ord('q'):
                    self.capture.release()
                    break
    
    
detector = Detector()
detector.run()

cv.destroyAllWindows()