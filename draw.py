#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib
# generate postscript output by default
# use before 'import matplotlib.pyplot as plt'
matplotlib.use('ps')

import math
import queue as qu
import copy as cp
import matplotlib.pyplot as plt
import networkx as net
import pygraphviz as pgv
from urllib import request


# def AGraph to draw : g
gDraw = pgv.AGraph(
    encoding = 'UTF-8',    # encode format
    rankdir  = 'LR',       # left to right, default'TB'
    directed = True        # directed graph
)

# inf
infinity = float("inf")

# dict of node label and node index
labelIndex = {} # str to int
indexLabel = {} # int to str

def floatEqual(x, y):
    return math.isclose(float(x), float(y))


class Graph:
    global infinity
    global gDraw
    global labelIndex
    global indexLabel

    def __init__(self, maps = [], nodenum = 0, edgenum = 0):
        self.map = maps    # The adjective matrix of graph
        self.nodenum = len(maps)
        self.edgenum = edgenum
        self.nodeTmp1 = {}
        self.nodeTmp2 = {}
        self.identifyQu = qu.Queue()
        self.mapTmp = maps

    def isOutRange(self, x):
        try :
            if x >= self.nodenum or x <= 0:
                raise  IndexError
        except IndexError:
            print("Index out of range!")
             
    def GetNodenum(self):
        self.nodenum = len(self.map)
        return self.nodenum

    def GetEdgenum(self):
        GetNodenum()
        self.edgenum = 0
        for i in range(self.nodenum):
            for j in range(self.nodenum):
                if math.isclose(float(self.map[i][j]), 0) or math.isclose(float(self.map[i][j]), 1):
                    self.edgenum = self.edgenum + 1
        return self.edgenum
     
    def InsertNode(self, node):
        for i in range(self.nodenum):
            self.map[i].append(infinity)
        self.nodenum = self.nodenum + 1
        ls = [infinity] * self.nodenum
        self.map.append(ls)
        gDraw.add_node(node, shape = 'circle')

    def InverseDict(self, nodeDict):
        dictTmp = {}
        for nodeNum in range(self.nodenum):
            if indexLabel[nodeNum] not in nodeDict:
                dictTmp[indexLabel[nodeNum]] = nodeNum
        return dictTmp
         
    def AddEdge(self, start, end, state):
        
        x = labelIndex[start]
        y = labelIndex[end]

        if self.map[x][y] is infinity:
            self.map[x][y] = state
            self.edgenum = self.edgenum + 1
            gDraw.add_edge(start, end, label = state)
    
    # delete node 
    def RemoveNode(self, delNode):
        nodeNum = labelIndex[delNode]
        for i in range(self.nodenum):
            if floatEqual(self.map[i][nodeNum], 0) or floatEqual(self.map[i][nodeNum], 1):
                self.RemoveEdge(indexLabel[i], indexLabel[nodeNum])
            if floatEqual(self.map[nodeNum][i], 0) or floatEqual(self.map[nodeNum][i], 1):
                self.RemoveEdge(indexLabel[nodeNum], indexLabel[i])
        for i in range(self.nodenum):
            self.map[i][nodeNum] = -1
            self.map[nodeNum][i] = -1

        # delete node in gDraw
        gDraw.remove_node(delNode)
        del indexLabel[nodeNum]
        del labelIndex[delNode]

    # merge node, used in simplifyEqual
    def MergeNode(self, merNode1, merNode2):
        nodeNum1 = labelIndex[merNode1]
        nodeNum2 = labelIndex[merNode2]
        if nodeNum1 > nodeNum2:
            tmp = nodeNum1
            nodeNum1 = nodeNum2
            nodeNum2 = tmp

        print('   Merge node:', indexLabel[nodeNum1], indexLabel[nodeNum2])
        # add edges which was not exsit in merNode1 but exsit in merNode2
        for i in range(self.nodenum):
            if floatEqual(self.map[i][nodeNum2], 0) or floatEqual(self.map[i][nodeNum2], 1):
                if floatEqual(self.map[i][nodeNum1], infinity):
                    self.map[i][nodeNum1] = self.map[i][nodeNum2]
                    gDraw.add_edge(indexLabel[i], indexLabel[nodeNum1], label = self.map[i][nodeNum1])
            if floatEqual(self.map[nodeNum2][i], 0) or floatEqual(self.map[nodeNum2][i], 1):
                if floatEqual(self.map[nodeNum1][i], infinity):
                    self.map[nodeNum1][i] = self.map[nodeNum2][i]
                    gDraw.add_edge(indexLabel[nodeNum1], indexLabel[i], label = self.map[nodeNum1][i])
        n = gDraw.get_node(indexLabel[nodeNum1])
        str = indexLabel[nodeNum1] + ',' + indexLabel[nodeNum2]
        n.attr['label'] = str
        self.RemoveNode(indexLabel[nodeNum2])
        

    def RemoveEdge(self, start, end):
        x = labelIndex[start]
        y = labelIndex[end]

        if math.isclose(float(self.map[x][y]), 0) or math.isclose(float(self.map[x][y]), 1):
            self.map[x][y] = -1
            self.edgenum = self.edgenum - 1
            #print("remove:", start, end)
            #gDraw.remove_edge(start, end)

    # identify nodes that can be reached from startNode
    def simplifyUseless_sub1(self, startNode):
        startOrd = labelIndex[startNode]
        for i in range(self.nodenum):
            if self.map[startOrd][i] is not infinity:
                node = indexLabel[i]
                if node not in self.nodeTmp1.keys():
                    self.nodeTmp1[node] = i
                    self.simplifyUseless_sub1(node)

    # identify nodes that can get to endNode
    def simplifyUseless_sub2(self,endNode):
        endOrd = labelIndex[endNode]

        for i in range(self.nodenum):
            if self.map[i][endOrd] is not infinity:
                node = indexLabel[i]
                if node not in self.nodeTmp2:
                    self.nodeTmp2[node] = i
                    self.simplifyUseless_sub2(node)

    # delete nodes that is useless
    def simplifyUseless(self, startNode, endNode):

        self.simplifyUseless_sub1(startNode)
        self.simplifyUseless_sub2(endNode)

        # Inverse the nodeTmp ton get the useless node
        self.nodeTmp1 = self.InverseDict(self.nodeTmp1)
        self.nodeTmp2 = self.InverseDict(self.nodeTmp2)

        # nodeTmp1: nodes that can't be reached from startNode
        # nodeTmp2: nodes that can't get to endNode
        for node in labelIndex:
            if node in self.nodeTmp1:
                if node in self.nodeTmp2:
                    #del self.nodeTmp1[node]
                    del self.nodeTmp2[node]

        print('1.Delete Useless Nodes:')
        
        # delete node in nodeTmp1
        for node in self.nodeTmp1:
            self.RemoveNode(node)
            print(node)

        # delete node in self.nodeTmp2
        for node in self.nodeTmp2:
            self.RemoveNode(node)
            print(node)


    # identify the equal nodes
    def FindEqual(self, x, y, state):
        for i in range(self.nodenum):
            if floatEqual(self.map[i][x], 0) or floatEqual(self.map[i][x], 1):
                for j in range(self.nodenum):
                    if floatEqual(self.map[j][y], self.map[i][x]) and not floatEqual(self.map[j][y], -1):
                        if(i>j):
                            
                            if(self.mapTmp[i][j] is 0):
                                self.mapTmp[i][j] = state + 1
                                tmp = list()
                                tmp.append(indexLabel[i])
                                tmp.append(indexLabel[j])
                                tmp.append(state+1)
                                self.identifyQu.put(tmp)
                        if(i<j):
                            if(self.mapTmp[j][i] is 0):
                                self.mapTmp[j][i] = state + 1
                                tmp = list()
                                tmp.append(indexLabel[i])
                                tmp.append(indexLabel[j])
                                tmp.append(state+1)
                                self.identifyQu.put(tmp)


    # delete equal nodes
    def simplifyEqual(self, startNode, endNode):
        self.mapTmp = cp.deepcopy(self.map)
        for i in range(len(self.mapTmp)):
            for j in range(len(self.mapTmp)):
                # exclude the deleted nodes which value is -1
                if not floatEqual(self.mapTmp[i][j],-1):
                    # init the value to 0
                    self.mapTmp[i][j] = 0
        endNum = labelIndex[endNode]
        for node in labelIndex:
            if labelIndex[node] is not endNum:
                tmp = list()
                tmp.append(node)
                tmp.append(endNode)
                tmp.append(1)
                self.identifyQu.put(tmp)
                # mark identifiable nodes first time
                if(labelIndex[node] > endNum):
                    self.mapTmp[labelIndex[node]][endNum] = 1
                else:
                    self.mapTmp[endNum][labelIndex[node]] = 1

        while(not self.identifyQu.empty()):
            tmp1 = self.identifyQu.get()
            i = labelIndex[tmp1[0]]
            j = labelIndex[tmp1[1]]
            state = int(tmp1[2])
            self.FindEqual(i, j, state)

        print('\n2.Merge equal nodes')
        for i in range(len(self.mapTmp)):
            for j in range(i):
                if i is not j:
                    if self.mapTmp[i][j] is 0:
                        self.MergeNode(indexLabel[i],indexLabel[j])


# set node shape
def SetNodeStyle(node,shape = 'circle'):
    global gDraw
    n = gDraw.get_node(node)
    n.attr['shape'] = shape


# Main Function
def DoExp():

    global infinity
    global gDraw
    global labelIndex
    global indexLabel

    # matrix G
    G = Graph()

    # set default layout of edges
    gDraw.edge_attr['color'] = 'black'
    # set label of AGraph
    gDraw.graph_attr['label'] = 'DFA'

    # open file
    fin = open("graph.in")
    print('      Loading data . . .')
    print('------------------------------')

    # input node num and edge(int)
    nodeNum = int(fin.readline().strip('\n'))
    print('nodeNum:',nodeNum)
    edgeNum = int(fin.readline().strip('\n'))
    print('edgeNum',edgeNum)

    # input node
    print('\n\nInput node:')
    for i in range(nodeNum):
        node = fin.readline().strip('\n')
        strTmp = 'No.'+str(i+1)+' node:'
        print(strTmp,node)
        labelIndex[str(node)] = int(i)
        indexLabel[int(i)] = str(node)
        G.InsertNode(node)

    # indicate startNode and endNode(string)
    startNode = fin.readline().strip('\n')
    endNode = fin.readline().strip('\n')
    print('Start node:', startNode)
    print('End node:', endNode)

    # init style of startNode and endNode in g
    gDraw.add_node('Start', shape='none')
    gDraw.add_edge('Start', startNode)
    SetNodeStyle(endNode,'doublecircle')

    # input edge('pi 1 pj' to stand for pi-1->pj)
    print('\n\nInput  qi 1 qj  to stand for (qi,1) = qj')
    for i in range(edgeNum):
        strTmp = 'No.'+str(i+1)+' edge:'
        edgeStr = fin.readline().strip('\n')
        print(strTmp, edgeStr)
        edgeStr = edgeStr.split(' ')
        G.AddEdge(edgeStr[0],edgeStr[2],edgeStr[1])

    print('------------------------------')
    print('   Data loading complete')

    # read complete. close file
    fin.close()

    # draw pic before simplify
    gDraw.layout(prog ='dot')
    gDraw.draw('before.png')
    print("DiGraph saved to 'before.png'\n\n")


    # simplify the AGraph g through #1 and #2
    # 1.identify the useless node
    G.simplifyUseless(startNode, endNode)

    # 2.identify the equal node
    G.simplifyEqual(startNode, endNode)


    # draw pic after simplify
    gDraw.layout(prog ='dot')
    gDraw.draw('after.png')
    print("\n\nDiGraph saved to 'after.png'")


if __name__ == '__main__':
    DoExp()
