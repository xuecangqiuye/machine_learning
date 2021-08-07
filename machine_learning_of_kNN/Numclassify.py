from numpy import *
import os
import operator


# kNN classify
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    a = tile(inX, (dataSetSize, 1))
    diffMat = a - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        a = classCount.get(voteIlabel, 0)
        classCount[voteIlabel] = a + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# Vectorization
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir(trainpath)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(trainpath + os.sep + fileStr + '.txt')
    testFileList = os.listdir(testpath)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(testpath + os.sep + fileStr + '.txt')

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print('the classifier came back with {0}, the real answer is {1}'.format(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1
        print('\nthe total number of errors is: {}'.format(errorCount))
        print('\nthe total error rate is: {}'.format((errorCount / float(mTest))))


trainpath = 'trainingDigits'
testpath = 'testDigits'

handwritingClassTest()
