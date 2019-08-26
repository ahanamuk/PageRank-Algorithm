#Sample input is as follows:
#rank([[1,3,4],[0,2,4],[3,6],[2,4,6],[5,8],[4,6,8],[0,7,9],[0,6,8],[2,9],[0,2,8]])
#Sample output is the following:
#[2, 6, 8, 0, 3, 9, 4, 5, 7, 1]
#The primary function is called rank. This function calls all other helper functions

import numpy as np
import math
import copy
    
def rank(links):
    #creates matrix with all zeroes
    transitionM = []
    for i in range(len(links)):
        transitionM.append([])
        for j in range(len(links)):
            transitionM[i].append(0)
    #puts in values for matrix A
    for i in range(len(links)):
        probability = 1/(len(links[i]))
        for j in links[i]:
            transitionM[j][i] = probability
    #change probability for all dangling nodes to 1/n, n = number of pages
    danglingNode(transitionM)
    #fill vector x with 1/n where n is # of pages
    n = len(transitionM)
    v = [1/n for i in range(len(transitionM[0]))]
    #multiply Av recursively
    oldV = matrixVectorMultiply(transitionM, v) #Av
    mult = np.matmul(transitionM, oldV) #AAv
    newV = np.ndarray.tolist(mult) #changes AAv to lists
    diff = True
    while diff == True:
        oldV = newV #AAv
        mult = np.matmul(transitionM, newV) #AAAv
        newV = np.ndarray.tolist(mult) #changes AAAv to lists
        diff = convergence(oldV, newV)
    rankedList = createRank(newV)
    return rankedList
    
def matrixVectorMultiply(A, v):
    newV = []
    #checks that the dimensions of A and v match
    if (len(A[0]) != len(v)):
        return False
    #multiplies Av and creates resulting vector "newV"
    for j in range(len(A[0])):
        sum = 0
        for i in range(len(v)):
            sum += v[i]*A[j][i]
        newV.append(sum)
    return newV
    
def isDanglingNode(col):
    #checks if column contains all zeroes
    for elem in range(len(col)):
        if elem != 0:
            return False
    return True
    
def danglingNode(transitionM):
    #transpose matrix to get columns
    transitionMT = np.ndarray.tolist(np.transpose(transitionM))
    #total number of pages
    n = len(transitionM)
    #loop through columns to check if it's a danging node
    for col in transitionMT:
        if isDanglingNode(col):
            #change all elements in column to 1/n
            for elem in col:
                elem = 1/n
    return np.ndarray.tolist(np.transpose(transitionMT))
    
def createRank(v):
    list = copy.deepcopy(v)
    rankedList = []
    #finds max element, appends to new list, removes element from original list
    while len(v) > 0:
        m = max(v)
        i = list.index(m)
        rankedList.append(i)
        v.pop(v.index(m))
    return rankedList
    
def convergence(oldV, newV):
    #checks if difference between all values is less than 0.001
    for i in range(len(oldV)):
        if abs(oldV[i]) - abs(newV[i]) >= 0.001:
            return True
    return False



