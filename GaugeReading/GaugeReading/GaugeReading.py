import os
import cv2
import imutils
import glob
import numpy as np

def getImages(fileDirectory, gray = False, edges = False):
    images = []
    for name in glob.glob(fileDirectory + "/*"):
        image = cv2.imread(name)
        if(gray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if(edges):
            image = cv2.Canny(image, 50, 420)
        images.append(image)
    return images

def rotateImage(image, degrees):
    (height, width) = image.shape[:2]
    center = (width / 2, height / 2)

    rotationMatrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotatedImage = cv2.warpAffine(image, rotationMatrix, (width, height))

    return rotatedImage

def getDistance(point1, point2):
        return np.sqrt((float(point1[0]) - float(point2[0]))**2 + (float(point1[1]) - float(point2[1]))**2)

def displayImage(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getLine(image, centerRangeRatio):

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaugeCenter = (image.shape[1] / 2, image.shape[0] / 2)
    gaugeRadius = image.shape[1] / 2

    thresh = 175
    maxvalue = 255

    th, dst2 = cv2.threshold(grayImage, thresh, maxvalue, cv2.THRESH_BINARY_INV)

    minLineLength = 10
    maxLineGap = 0
    lines = cv2.HoughLinesP(image = dst2, rho=3, theta=np.pi/180, threshold=100, minLineLength=minLineLength, maxLineGap=0)

    finalLines = []
    for line in lines:
        line = line[0]
        x1, y1, x2, y2 = line
        point1 = (x1, y1)
        point2 = (x2, y2)
        if(getDistance(gaugeCenter, point1) < gaugeRadius * centerRangeRatio or getDistance(gaugeCenter, point2) < gaugeRadius * centerRangeRatio):
            finalLines.append(line)
    finalLine = max(finalLines, key = lineLength)
    print finalLine
    return finalLine;

def lineLength(line):
    x1, y1, x2, y2 = line
    point1 = (x1, y1)
    point2 = (x2, y2)
    return getDistance(point1, point2)

def getCircle(image):
    (height, width) = image.shape[:2]
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayImage, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, 0, int(height * 0.48))
    circle = circles[0][0]
    (x, y, r) = circle
    
    mask = np.ones((height, width), np.uint8)
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    maskedData = cv2.bitwise_and(image, image, mask=mask)
    _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x1, y1, h, w = cv2.boundingRect(contours[0])
    crop = maskedData[y1:y1+h, x1:x1+w]

    return (circle, crop)

def getLineAngle(image, line):
    centerX = float(image.shape[1] / 2)
    centerY = float(image.shape[0] / 2)

    x1, y1, x2, y2 = line

    if(getDistance((centerX, centerY), (x1, y1)) > getDistance((centerX, centerY), (x2, y2))):
        xLength = x1 - centerX
        yLength = y1 - centerY
    else:
        xLength = x2 - centerX
        yLength = y2 - centerY

    angle = np.arctan(float(abs(yLength)) / float(abs(xLength) + 0.000000000000000000000001))
    angle = np.rad2deg(angle)
    #topleft corner is position of 0, 0 so must be accounted for
    if(xLength > 0 and yLength < 0): #Quadrant 1
        print(angle)
        angle = 270 - angle
        print("1")
    elif(xLength <= 0 and yLength <= 0): #Quadrant 2
        print("2")
        print(angle)
        angle = 90 + angle
    elif(xLength < 0 and yLength > 0): #Quadrant 3
        print(angle)
        angle = 90 - angle
        print("3")
    elif(xLength > 0 and yLength >= 0): #Quadrant 4
        print(angle)
        angle = 270 + angle
        print("4")

    print("angle:", angle, xLength, yLength)
    return angle


path = os.path.dirname(os.path.realpath(__file__))

image = getImages(path + "/Images")[1]
displayAllSteps = True

#Angle where gauge reads zero
angleZero = 45.0
#Angle where gauge is at max
angleMax = 315.0
#Max value gauge reads
largestValue = 1.0

valuePerDegree =  largestValue / (angleMax - angleZero)
print("PerDeg", valuePerDegree)
for angle in range(0, -360, -5):

    rotatedImage = rotateImage(image, angle)

    circle, circleImage = getCircle(rotatedImage);
    mainLine = getLine(circleImage, 0.5)
    if(displayAllSteps):
        displayImage(rotatedImage)
        displayImage(circleImage)

    angle = getLineAngle(circleImage, mainLine)
    print("Gauge Reading:", (angle-angleZero) * valuePerDegree)
    cv2.line(circleImage, tuple(mainLine[:2]), tuple(mainLine[2:]), (0, 0, 255), 3, cv2.LINE_AA)
    displayImage(circleImage)
