import os
import time
import numpy as np
import imutils
import cv2
import glob

def getImages(fileDirectory, edges = False):
    images = []
    for name in glob.glob(fileDirectory + "/*"):
        image = cv2.imread(name)
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

def multiscaleMatch(template, grayImage):
    (templateH, templateW) = template.shape[:2]
    found = None

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        #shape[1] == imageWidth
        resized = imutils.resize(grayImage, width = int(grayImage.shape[1] * scale))
        #Used to scale image back to original size
        imageRatio = grayImage.shape[1] / float(resized.shape[1])

        #Don't check if image is smaller then the template
        if(resized.shape[0] < templateH or resized.shape[1] < templateW):
            break

        edgedImage = cv2.Canny(resized, 50, 420)
        matchResult = cv2.matchTemplate(edgedImage, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(matchResult)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, imageRatio, template)

    return found

def drawBox(image, locationInformation):
    (maxVal, maxLoc, imageRatio, template) = locationInformation
    (templateH, templateW) = template.shape[:2]

    topLeft = (int(maxLoc[0] * imageRatio), int(maxLoc[1] * imageRatio))
    bottomRight = (int((maxLoc[0] + templateW) * imageRatio), int((maxLoc[1] + templateH) * imageRatio))

    cv2.rectangle(image, topLeft, bottomRight, (0, 0, 225), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Path to this .py file directory
path = os.path.dirname(os.path.realpath(__file__))
rotations = 5

templates = getImages(path + "/Templates", edges=True);
grayImages = getImages(path + "/Images")
timeSpent = []

for image in grayImages:
    startTime = time.time()
    info = []
    for template in templates:
        for i in range(rotations):
            #Rotating an image degrades its quality slightly
            rotatedTemplate = rotateImage(template, 360/rotations * i)
            info.append(multiscaleMatch(rotatedTemplate, image))

    bestInfo = max(info)
    timeSpent.append(time.time() - startTime)
    cv2.imshow("Template", bestInfo[3])
    print("Best Score", bestInfo[0])
    drawBox(image, bestInfo)

print(timeSpent)
print("Avg Time:", sum(timeSpent) / len(timeSpent))

