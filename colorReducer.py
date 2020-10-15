#!/usr/bin/env python3

import numpy as np
from PIL import Image as PILimage
import random
import math
from tabulate import tabulate


class Cluster:
    def __init__(self):
        self.stillSame = True
        self.colorVector = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    def compare(self, vector):
        return math.sqrt((self.colorVector[0] - vector[0])**2 + (self.colorVector[1] - vector[1])**2 + (self.colorVector[2] - vector[2])**2)

    def attribute(self, meanVector):
        for i in [0, 1, 2]:
            self.stillSame = self.stillSame and self.colorVector[i] == meanVector[i]

        self.colorVector = meanVector

    def beginCycle(self):
        self.stillSame = True

    def __str__(self):
        return self.colorVector.__str__()


def clustersToNpArray(clusters):
    # Table of float
    toReturn = np.zeros(shape=(len(clusters), 3))
    for i in range(len(clusters)):
        toReturn[i] = clusters[i].colorVector
    return toReturn


def checkCluster(clusters):
    for cluster in clusters:
        if not cluster.stillSame:
            return False
    return True


def indexNearestCluster(clusters, vector):
    lowest = 0
    lowestDistance = clusters[0].compare(vector)

    for i in range(1, len(clusters)):
        currentDistance = clusters[i].compare(vector)
        if currentDistance <= lowestDistance:
            lowest = i
            lowestDistance = currentDistance

    return lowest


# Use k-mean clustering Algorithm
def imageToColors(colorNumber, path, maxCycle=100, debug=False):
    clusters = []
    for i in range(colorNumber):
        clusters.append(Cluster())

    meanVectors = np.zeros(shape=(len(clusters), 3), dtype=int)
    # Number of pixel attributed for each cluster
    numberAttributed = np.zeros(shape=(len(clusters), 1), dtype=int)

    data = np.array(PILimage.open(path))

    if debug:
        test = np.zeros(shape=(len(data), len(data[0]), 3), dtype=np.uint8)

    cycle = 0

    # Do while
    while True:
        for i in range(len(clusters)):
            clusters[i].beginCycle()

        for xPos in range(len(data)):
            y = data[xPos]
            for yPos in range(len(data[xPos])):
                # For each pixels
                index = indexNearestCluster(clusters, y[yPos])
                meanVectors[index] += y[yPos]
                numberAttributed[index] += 1
                if debug:
                    test[xPos][yPos] = [index / colorNumber * 255, index / colorNumber * 255, index / colorNumber * 255]

        for i in range(len(clusters)):
            if numberAttributed[i] != 0:
                clusters[i].attribute(np.array(meanVectors[i] / numberAttributed[i]))

            meanVectors[i] = [0, 0, 0]
            numberAttributed[i] = 0

        # Exit statement
        if checkCluster(clusters) or maxCycle <= cycle:
            if debug:
                with open('test/table.txt', 'w') as f:
                    f.write(tabulate(test))

                image = PILimage.fromarray(test, 'RGB')
                image.show()
            break
        else:
            cycle += 1
            if debug:
                if cycle % 10 == 0:
                    print("cycle :" + cycle.__str__())

    return clustersToNpArray(clusters)

# Take colors choosed (0->255 X 3) to normalize it (-1 -> 1 X 3)
def normalizeData(colorNumber, path, maxCycle=100, debug=False):
    return (imageToColors(colorNumber, path, maxCycle, debug) - 128) / 128
