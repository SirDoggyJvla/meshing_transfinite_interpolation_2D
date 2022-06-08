# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:42:30 2022

@author: utilisateur
"""
from ImportAllS import *


class ListAround():
    # Lists Nodes
    def generateListNodesNodes(self):
        """
        --- Creates the lists of every Nodes linked by an edge with the Nodes, for every Nodes
        A.generateListNodesNodes(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : list
            - first term is first Node, last term is last Node
            - rotation rule start: left
        """
        self.listNodesNodes = []
        for i in range(0,self.N):
            temp = []
            pos1 = 0
            pos2 = 0
            for j in range(0,len(self.edgesHor)):
                if self.edgesHor[j,1] == i:
                    temp.append(self.edgesHor[j,0])
                    pos1 = j + 1
            for j in range(0,len(self.edgesVer)):
                if self.edgesVer[j,1] == i:
                    temp.append(self.edgesVer[j,0])
                    pos2 = j + 1
            for j in range(pos1,len(self.edgesHor)):
                if self.edgesHor[j,0] == i:
                    temp.append(self.edgesHor[j,1])
            for j in range(pos2,len(self.edgesVer)):
                if self.edgesVer[j,0] == i:
                    temp.append(self.edgesVer[j,1])
            self.listNodesNodes.append(temp)
        return self.listNodesNodes
    
    def generateListNodesEdges(self):
        """
        --- Creates the lists of every Edges linked to the Nodes, for every Nodes
        A.generateListNodesEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : list
            - first term is first Node, last term is last Node
            - rotation rule start: left
        """
        self.listNodesEdges = []
        for i in range(0,len(self.X_ref)):
            temp = []
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,1] == i:
                    temp.append(j)
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,1] == i:
                    temp.append(j)
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,0] == i:
                    temp.append(j)
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,0] == i:
                    temp.append(j)
            self.listNodesEdges.append(temp)
        return self.listNodesEdges
        
    def generateListNodesElements(self):
        """
        --- Creates the lists of every elements surrounding the Nodes, for every Nodes
        A.generateListNodesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesElements        : list
            - first term is first Node, last term is last Node
            - rotation rule start: bottom left
        """
        self.listNodesElem = []
        for i in range(0,len(self.X_ref)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.elem[j,2] == i:
                    temp.append(j)
                if self.elem[j,3] == i:
                    temp.append(j)
                if self.elem[j,0] == i:
                    temp.append(j)
                if self.elem[j,1] == i:
                    temp.append(j)
            if len(temp) == 4:
                a = temp[2]
                b = temp[3]
                temp[2] = b
                temp[3] = a
            self.listNodesElem.append(temp)
        return self.listNodesElem


    # Lists Edges
    def generateListEdgesElements(self):
        """
        --- Creates the lists of every elements surrounding the edges, for every edges
        A.generateListEdgesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listEdgesElements      : list
            - first term is first edge, last term is last edge, with the horizontal and vertical edges keeping the same order as self.edges
            - rotation rule start: bottom left
        """
        self.listEdgesElements = []
        for i in range(0,len(self.edgesHor)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.edgesHor[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,3] and self.edgesHor[i,1] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesHor[i,1] == self.elem[j,3]:
                    temp.append(j)
                    
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesHor[i,1] == self.elem[j,0]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,0] and self.edgesHor[i,1] == self.elem[j,1]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,1]:
                    temp.append(j)    
            self.listEdgesElements.append(temp)
        
        for i in range(0,len(self.edgesVer)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.edgesVer[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesVer[i,0] == self.elem[j,3]:
                    temp.append(j)
                if self.edgesVer[i,0] == self.elem[j,0] and self.edgesVer[i,1] == self.elem[j,3]:
                    temp.append(j)
                if self.edgesVer[i,1] == self.elem[j,0]:
                    temp.append(j)

            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesVer[i,1] == self.elem[j,1]:
                    temp.append(j)
                if self.edgesVer[i,1] == self.elem[j,2] and self.edgesVer[i,0] == self.elem[j,1]:
                    temp.append(j)
            self.listEdgesElements.append(temp)
        return self.listEdgesElements
    
    
    # Lists Elements
    def generateListElementsEdges(self):
        """
        --- Creates the lists of every edges surrounding the elements, for every elements
        A.generateListElementsEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsEdges      : list
            - first term is first element, last term is last element
            - rotation rule start: left
        """
        self.listElementsEdges = []
        for i in range(0,len(self.elem)):
            temp = []
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,3]:
                    temp.append(j)
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,1]:
                    temp.append(j)
                if self.edges[j,0] == self.elem[i,1] and self.edges[j,1] == self.elem[i,2]:
                    temp.append(j)
            for j in range(0,len(self.edges)):
                if self.edges[j,1] == self.elem[i,2] and self.edges[j,0] == self.elem[i,3]:
                    temp.append(j)
            self.listElementsEdges.append(temp)
        self.listElementsEdges = np.array(self.listElementsEdges)
        return self.listElementsEdges

    def generateListElementsElements(self):
        """
        --- Creates the lists of every elements surrounding the elements, for every elements
        A.generateListElementsElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsElements      : list
            - first term is first element, last term is last element
            - rotation rule start: left
        """
        self.listElementsElements = []
        for i in range(0,len(self.elem)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,1] and self.elem[i,3] == self.elem[j,2]:
                    temp.append(j)
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.elem[i,0] == self.elem[j,3] and self.elem[i,1] == self.elem[j,2]:
                    temp.append(j)
                if self.elem[i,1] == self.elem[j,3]:
                    temp.append(j)
                if self.elem[i,1] == self.elem[j,0] and self.elem[i,2] == self.elem[j,3]:
                    temp.append(j)
                if self.elem[i,2] == self.elem[j,0]:
                    temp.append(j)
            for j in range(len(self.elem)-1,-1,-1):
                if self.elem[i,2] == self.elem[j,1] and self.elem[i,3] == self.elem[j,0]:
                    temp.append(j)                    
                if self.elem[i,3] == self.elem[j,1]:
                    temp.append(j)
            self.listElementsElements.append(temp)
        return self.listElementsElements

    # List main
    def listAround(self,forEach=None,get=None):
        """
        --- Gives a list of nodes, elements or edges next to nodes, elements or edges
            They are listed by following the anti-clockwise:
                example:
                    
                    2---3
                    | 0 |   ===> listElementsNodes = [[0,1,3,2]]
                    0---1
                  
                rule for the order in the list: first is left object, 
                or bottom left object if no direct object at left
                    
                  ↙ ←----←----←------←-------←----- ↖
                ↙                                     ↖
               ↓    top left     top    top right      ↑
                                o---o                  |
               start  left      |   |   right          |
               |                o---o                  ↑
               ↓  bottom left   bottom  bottom right   |
                ↘  alt start                         ↗
                  ↘ -→--------→---------→--------→ ↗
                    
        A.listAround(forEach=,get=)
        
        -- Input                will give an array (L,l) with every chosen 'get' around that nodes
        forEach         : str       
            -'Nodes'      L = N
            -'Elements'   L = N_el
            -'Edges'      L = N_ed
        get             : str
            -'Nodes'      
            -'Elements'   l depends on the combinations made between 'forEach' and 'get'
            -'Edges'      
        OR
        give='Nodes',get=None   will give an array (N,2) of every coordinates of the Nodes
        
        
        -- Ouput
        listResult      : array (L,l)
        """
        # print(forEach)
        # print(get)
        listResult = None
        if forEach == 'Nodes':
            if get == 'Nodes': listResult = self.generateListNodesNodes()
            elif get == 'Edges': listResult = self.generateListNodesEdges()
            elif get == 'Elements': listResult = self.generateListNodesElements()
            elif get == None: listResult = self.X_ref
            
        elif forEach == 'Edges': 
            if get == 'Nodes': listResult = self.edges
            elif get == 'Edges': listResult = None
            elif get == 'Elements': listResult = self.generateListEdgesElements()
            
        elif forEach == 'Elements':
            if get == 'Nodes': listResult = self.elem
            elif get == 'Edges': listResult = self.generateListElementsEdges()
            elif get == 'Elements': listResult = self.generateListElementsElements()
        return listResult