import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import *
import os
import sudukoSolver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pathImage = "Resources/3.jpg"
heightImg = 450
widthImg = 450
model = load_model("myModel.h5")

##Preprocessing image
img = cv2.imread(pathImage)
img = cv2.resize(img,(heightImg,widthImg))
imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)
imgThreshold = preProcess(img)

##Finding countours
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours,-1,(0,255,0),3)

##Find biggest contour
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour,biggest,-1,(0,0,255),20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix,(widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    ##split into 81 cells
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    cv2.imshow('boxes',boxes[0])
    #numbers()
    numbers = getPrediction(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255,0,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers>0,0,1)

    ##find actual solution
    board = np.array_split(numbers,9)
    try:
        sudukoSolver.solve(board)
    except:
        pass
    flatList = []
    for subList in board:
        for item in subList:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)

    ##overlay the solution
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)

    imageArrayContours = ([img, imgThreshold,imgBigContour])
    imageArraySolution = ([imgDetectedDigits, imgSolvedDigits,inv_perspective])
    stackedImageContours = stackImages(imageArrayContours, 1)
    stackedImageSolution = stackImages(imageArraySolution, 1)
    cv2.imshow('Stacked Image Contours', stackedImageContours)
    cv2.imshow('Stacked Image Solution',stackedImageSolution)
else:
    print("Sudoku not found")
cv2.waitKey(0)