from __future__ import print_function
from operator import truediv
import cv2 as cv
import argparse
import numpy as np
import pyautogui
print(cv.__version__)


# detects if point c is to the counter-clockwise of the line segment formed by points a and b
def isCounterClockwise(a, b, c):
    return ((b["X"] - a["X"])*(c["Y"] - a["Y"]) - (b["Y"] - a["Y"])*(c["X"] - a["X"])) > 0

backSub = cv.createBackgroundSubtractorKNN()
#LK parameters
lkParams = dict(winSize  = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
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
    type = 0 # 0 = color, 1 = background subtraction, 2 = KLT
    currentDirection = None
    lastFrame = None

    playGame = False

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

        # set up pixel locations in each quadrant
        for x in range(0,self.width):
            for y in range(0,self.height):
                quad = self.getQuadrant(x, y)

                if quad == "left":
                    # add index to upper triangle indices
                    self.left.append((y, x))
                elif quad == "right":
                    # add index to lower triangle indices
                    self.right.append((y, x))
                elif quad == "bot":
                    # add index to right triangle indices
                    self.bot.append((y, x))
                elif quad == "top":
                    # add index to left triangle indices
                    self.top.append((y, x))
                
        self.top = np.array(self.top)
        self.bot = np.array(self.bot)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
    
    # returns the quadrant of the point, left, right, top, bottom
    def getQuadrant(self, x, y):
            leftLine = True
            rightLine = True

            leftLine = isCounterClockwise({"X": 0, "Y": 0}, {"X": self.width, "Y": self.height}, {"X": x, "Y": y})
        
            rightLine = isCounterClockwise({"X": self.width, "Y": 0}, {"X": 0, "Y": self.height}, {"X": x, "Y": y})
            # print(x, y, top, bottom)
            if leftLine and rightLine:
                return "left"
            elif not leftLine and not rightLine:
                return "right"
            elif leftLine and not rightLine:
                return "bot"
            elif not leftLine and rightLine:
                return "top"

    def updateTestImage(self):
        # everything below y = 200 is the colorLower
        # everything above y = 200 is the colorUpper
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

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
           
            
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
            if self.type == 1:
                # background subtraction
                mask = backSub.apply(frame)
            elif self.type == 0:
                # color tracking
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask = cv.inRange(hsv, self.colorLower, self.colorUpper)
                if self.lastClick is not None:
                    cv.circle(frame, self.lastClick, 5, (0, 0, 255), -1)
            else:
                # KLT tracking
                if self.lastClick is not None:
                    lastPoints = np.array([[[self.lastClick[0], self.lastClick[1]]]], dtype=np.float32)
                    lastGray = cv.cvtColor(self.lastFrame, cv.COLOR_BGR2GRAY)
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    nextPoints, st, err = cv.calcOpticalFlowPyrLK(lastGray, gray, lastPoints, None, **lkParams)

                    if nextPoints is not None:
                        matchingNew = nextPoints[st==1]
                        matchingOld = lastPoints[st==1]
                    #put circles where tracked points are
                    for i, (new, _) in enumerate(zip(matchingNew, matchingOld)):
                        a, b = new.ravel()
                        frame = cv.circle(frame, (int(a), int(b)), 5, (0,0,255), -1)
            

            if self.currentDirection is not None:
                cv.putText(frame, self.currentDirection, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            

            if self.mode == 0:
                cv.imshow('detector', frame)
            elif self.mode == 1:
                # draw a red line from the upper-left corner to the lower-right corner
                cv.line(mask, (0, 0), (self.width,self.height), (255, 0, 0), 3)
                # draw a red line from the upper-right corner to the lower-left corner
                cv.line(mask, (self.width, 0), (0,self.height), (255, 0, 0), 3)
                cv.imshow('detector', mask)
            else:
                # combine the mask with the frame
                res = cv.bitwise_and(frame, frame, mask=mask)
                if self.currentDirection is not None:
                    cv.putText(res, self.currentDirection, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('detector', res)

            cv.imshow('HSV Range', self.testImg)
            
            if i % 5 == 0:
                dir = None
                if self.type == 2:
                    # KLT tracking update current direction
                    if self.lastClick is not None:
                        dir = self.getQuadrant(self.lastClick[0], self.lastClick[1]) # potentially use different x,y
                else:
                    # track quadrant based on the max value of each quadrant
                    topDiff = mask[self.top[:, 0], self.top[:, 1]]
                    bottomDiff = mask[self.bot[:, 0], self.bot[:, 1]]
                    leftDiff = mask[self.left[:, 0], self.left[:, 1]]
                    rightDiff = mask[self.right[:, 0], self.right[:, 1]]
                    
                    # #send that key to pygame
                    direction_scores = {sum(topDiff): "up", sum(rightDiff): "right", sum(bottomDiff): "down", sum(leftDiff): "left"}
                    dir = direction_scores.get(max(direction_scores))
                
                if self.playGame and dir is not self.currentDirection:
                    self.currentDirection = dir
                    pyautogui.press(self.currentDirection)
                else:
                    self.currentDirection = dir

            i = i + 1
            self.lastFrame = frame.copy()
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
                    if not self.playGame:
                        print("Play!")
                        self.playGame = True
                        
                elif keyboard == ord('/'):
                    if self.playGame:
                        print("Stop!")
                        self.playGame = False
                elif keyboard == ord('b'):
                    self.type = (self.type + 1) % 3
                # if the 'q' key is pressed
                elif keyboard == ord('q'):
                    self.capture.release()
                    break
    

if __name__ == "__main__":
    detector = Detector()
    detector.run()

    cv.destroyAllWindows()