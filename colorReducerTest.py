import colorReducer as crt
import numpy as np
from PIL import Image as PILimage
from tabulate import tabulate

path = 'test/koala.jpg'
colorNumber = 10

data = np.array(PILimage.open(path))

result = crt.imageToColors(colorNumber, path, 10, True)
print(result)

imageDataResult = np.zeros(shape=(len(data), len(data[0]), len(data[0][0])), dtype=np.uint8)

clusters = []
for i in range(colorNumber):
    clusters.append(crt.Cluster())
    clusters[i].attribute(result[i])

for xPos in range(len(data)):
    y = data[xPos]
    for yPos in range(len(data[xPos])):
        index = crt.indexNearCluster(clusters, y[yPos])
        imageDataResult[xPos][yPos] = clusters[index].colorVector

with open('test/image.txt', 'w') as f:
    f.write(tabulate(imageDataResult))

image = PILimage.fromarray(imageDataResult, 'RGB')
image.show()
image.save("test/colorReductionImage.png", "PNG")


