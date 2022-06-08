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
        --- Creates an array of every Nodes linked by an edge with the Nodes, for every Nodes
        A.generateListNodesNodes(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : array (N,4)    
            - first term is first Node, last term is last Node
            - rotation rule start: left
            means
                k4
                |
            k1--k0--k3   ===>    [[k1,k2,k3,k4]     <-- k0
                |                 ...
                k2                [..,..,..,..]]]   <-- kn
            - if k doesn't exist then = -1
                k4
                |
                k0--k3   ===>    [[-1,-1,k3,k4]     <-- k0
                                  ...
                                  [..,..,..,..]]    <-- kn
        """
        self.listNodesNodes = np.zeros((self.N,4))
        for i in range(0,self.N):
            pos1 = 0
            pos2 = 0
            for j in range(0,len(self.edgesHor)):
                if self.edgesHor[j,1] == i:
                    self.listNodesNodes[i,0] = self.edgesHor[j,0]
                    pos1 = j + 1
                    break
                else:
                    self.listNodesNodes[i,0] = None
            for j in range(0,len(self.edgesVer)):
                if self.edgesVer[j,1] == i:
                    self.listNodesNodes[i,1] = self.edgesVer[j,0]
                    pos2 = j + 1
                    break
                else:
                    self.listNodesNodes[i,1] = None
            for j in range(pos1,len(self.edgesHor)):
                if self.edgesHor[j,0] == i:
                    self.listNodesNodes[i,2] = self.edgesHor[j,1]
                    break
                else:
                    self.listNodesNodes[i,2] = None
            for j in range(pos2,len(self.edgesVer)):
                if self.edgesVer[j,0] == i:
                    self.listNodesNodes[i,3] = self.edgesVer[j,1]
                    break
                else:
                    self.listNodesNodes[i,3] = None
        self.listNodesNodes[self.N-1,2] = None
        self.listNodesNodes[self.N-1,3] = None
        return self.listNodesNodes
    
    def generateListNodesEdges(self):
        """
        --- Creates an array of every Edges linked to the Nodes, for every Nodes
        A.generateListNodesEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : array (N,4)
            - first term is first Node, last term is last Node
            - rotation rule start: left
            follows same reading principle as A.listNodesNodes()
        """
        self.listNodesEdges = np.zeros((self.N,4))
        for i in range(0,len(self.X_ref)):
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,1] == i:
                    self.listNodesEdges[i,0] = j
                    break
                else:
                    self.listNodesEdges[i,0] = None
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,1] == i:
                    self.listNodesEdges[i,1] = j
                    break
                else:
                    self.listNodesEdges[i,1] = None
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,0] == i:
                    self.listNodesEdges[i,2] = j
                    break
                else:
                    self.listNodesEdges[i,2] = None
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,0] == i:
                    self.listNodesEdges[i,3] = j
                    break
                else:
                    self.listNodesEdges[i,3] = None
        return self.listNodesEdges
        
    def generateListNodesElements(self):
        """
        --- Creates an array of every elements surrounding the Nodes, for every Nodes
        A.generateListNodesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesElements        : array (N,4)
            - first term is first Node, last term is last Node
            - rotation rule start: bottom left
            follows same principle reading as before, 
            but here it starts BOTTOM LEFT 
            because no element is at direct left
        """
        self.listNodesElem = np.zeros((self.N,4))
        for i in range(0,len(self.X_ref)):
            for j in range(0,len(self.elem)):
                if self.elem[j,2] == i:
                    self.listNodesElem[i,0] = j
                    break
                else:
                    self.listNodesElem[i,0] = None
            for j in range(0,len(self.elem)):
                if self.elem[j,3] == i:
                    self.listNodesElem[i,1] = j
                    break
                else:
                    self.listNodesElem[i,1] = None
            for j in range(0,len(self.elem)):
                if self.elem[j,0] == i:
                    self.listNodesElem[i,2] = j
                    break
                else:
                    self.listNodesElem[i,2] = None
            for j in range(0,len(self.elem)):
                if self.elem[j,1] == i:
                    self.listNodesElem[i,3] = j
                    break
                else:
                    self.listNodesElem[i,3] = None
        return self.listNodesElem


    # Lists Edges
    def generateListEdgesElements(self):
        """
        --- Creates an array of every elements surrounding the edges, for every edges
        A.generateListEdgesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listEdgesElements      : array (N_ed,6)
            - first term is first edge, last term is last edge, 
              with the horizontal and vertical edges keeping the same order as self.edges in the array
            - rotation rule start: bottom left 
            means
            horizontal:
            o   o
          5 | 4 | 3
        o---o-k0o---o ===>  [[0,1,2,3,4,5]  <--- k0
          0 | 1 | 2         ...
            o   o            [.,.,.,.,.,.]] <--- kn
            vertical:
            o
          4 | 3
        o---o---o
          5 k0 2    ===>    [[0,1,2,3,4,5]  <--- k0
        o---o---o           ...
          0 | 1              [.,.,.,.,.,.]] <--- kn
            o
        """
        self.listEdgesElements = np.zeros((self.N_ed,6))
        # Horizontal
        for i in range(0,len(self.edgesHor)):
            for j in range(0,len(self.elem)):
                if self.edgesHor[i,0] == self.elem[j,2]:
                    self.listEdgesElements[i,0] = j
                    break
                else:
                    self.listEdgesElements[i,0] = None
            for j in range(0,len(self.elem)):
                if self.edgesHor[i,0] == self.elem[j,3] and self.edgesHor[i,1] == self.elem[j,2]:
                    self.listEdgesElements[i,1] = j
                    break
                else:
                    self.listEdgesElements[i,1] = None
            for j in range(0,len(self.elem)):
                if self.edgesHor[i,1] == self.elem[j,3]:
                    self.listEdgesElements[i,2] = j
                    break
                else:
                    self.listEdgesElements[i,2] = None
                    
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesHor[i,1] == self.elem[j,0]:
                    self.listEdgesElements[i,3] = j
                    break
                else:
                    self.listEdgesElements[i,3] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesHor[i,0] == self.elem[j,0] and self.edgesHor[i,1] == self.elem[j,1]:
                    self.listEdgesElements[i,4] = j
                    break
                else:
                    self.listEdgesElements[i,4] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesHor[i,0] == self.elem[j,1]:
                    self.listEdgesElements[i,5] = j
                    break
                else:
                    self.listEdgesElements[i,5] = None
        # Vertical
        for i in range(len(self.edgesHor),len(self.edgesHor)+len(self.edgesVer)):
            d = i - len(self.edgesHor)
            for j in range(0,len(self.elem)):
                if self.edgesVer[d,0] == self.elem[j,2]:
                    self.listEdgesElements[i,0] = j
                    break
                else:
                    self.listEdgesElements[i,0] = None
            for j in range(0,len(self.elem)):
                if self.edgesVer[d,0] == self.elem[j,3]:
                    self.listEdgesElements[i,1] = j
                    break
                else:
                    self.listEdgesElements[i,1] = None
            for j in range(0,len(self.elem)):
                if self.edgesVer[d,0] == self.elem[j,0] and self.edgesVer[d,1] == self.elem[j,3]:
                    self.listEdgesElements[i,2] = j
                    break
                else:
                    self.listEdgesElements[i,2] = None
            for j in range(0,len(self.elem)):
                if self.edgesVer[d,1] == self.elem[j,0]:
                    self.listEdgesElements[i,3] = j
                    break
                else:
                    self.listEdgesElements[i,3] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesVer[d,1] == self.elem[j,1]:
                    self.listEdgesElements[i,4] = j
                    break
                else:
                    self.listEdgesElements[i,4] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesVer[d,1] == self.elem[j,2] and self.edgesVer[d,0] == self.elem[j,1]:
                    self.listEdgesElements[i,5] = j
                    break
                else:
                    self.listEdgesElements[i,5] = None
        return self.listEdgesElements
    
    
    # Lists Elements
    def generateListElementsEdges(self):
        """
        --- Creates an array of every edges surrounding the elements, for every elements
        A.generateListElementsEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsEdges      : array (N_el,4)
            - first term is first element, last term is last element
            - rotation rule start: left
            mean
            o-3--o
            0 k0 2    ===>   [[0,1,2,3]  <--- k0
            o-1--o            ...
                              [.,.,.,.]] <--- kn
        """
        self.listElementsEdges = np.zeros((self.N_el,4))
        for i in range(0,len(self.elem)):
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,3]:
                    self.listElementsEdges[i,0] = j
                    break
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,1]:
                    self.listElementsEdges[i,1] = j
                    break
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,1] and self.edges[j,1] == self.elem[i,2]:
                    self.listElementsEdges[i,2] = j
                    break
            for j in range(0,len(self.edges)):
                if self.edges[j,1] == self.elem[i,2] and self.edges[j,0] == self.elem[i,3]:
                    self.listElementsEdges[i,3] = j
                    break
        self.listElementsEdges = self.listElementsEdges.astype(int)
        return self.listElementsEdges

    def generateListElementsElements(self):
        """
        --- Creates an array of every elements surrounding the elements, for every elements
        A.generateListElementsElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsElements      : list      a list
            - first term is first element, last term is last element
            - rotation rule start: left
            means
            o---o---o---o
            | k8| k7| k6|
            o---o---o---o
            | k1| k0| k5|   ===>   [[k1,k2,k3,k4,k5,k6,k7,k8]   <--- k0
            o---o---o---o           ...
            | k2| k3| k4|           [..,..,..,..,..,..,..,..]]  <--- kn
            o---o---o---o
        """
        self.listElementsElements = np.zeros((self.N_el,8))
        for i in range(0,len(self.elem)):
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,1] and self.elem[i,3] == self.elem[j,2]:
                    self.listElementsElements[i,0] = j
                    break
                else:
                    self.listElementsElements[i,0] = None
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,2]:
                    self.listElementsElements[i,1] = j
                    break
                else:
                    self.listElementsElements[i,1] = None
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,3] and self.elem[i,1] == self.elem[j,2]:
                    self.listElementsElements[i,2] = j
                    break
                else:
                    self.listElementsElements[i,2] = None
            for j in range(0,len(self.elem)):
                if self.elem[i,1] == self.elem[j,3]:
                    self.listElementsElements[i,3] = j
                    break
                else:
                    self.listElementsElements[i,3] = None
            for j in range(0,len(self.elem)):
                if self.elem[i,1] == self.elem[j,0] and self.elem[i,2] == self.elem[j,3]:
                    self.listElementsElements[i,4] = j
                    break
                else:
                    self.listElementsElements[i,4] = None
            for j in range(0,len(self.elem)):
                if self.elem[i,2] == self.elem[j,0]:
                    self.listElementsElements[i,5] = j
                    break
                else:
                    self.listElementsElements[i,5] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.elem[i,2] == self.elem[j,1] and self.elem[i,3] == self.elem[j,0]:
                    self.listElementsElements[i,6] = j
                    break
                else:
                    self.listElementsElements[i,6] = None
            for j in range(len(self.elem)-1,-1,-1):
                if self.elem[i,3] == self.elem[j,1]:
                    self.listElementsElements[i,7] = j
                    break
                else:
                    self.listElementsElements[i,7] = None
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