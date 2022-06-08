"""
S. Cadet
9/05/2022
-- Tools MeshCPU Mesh Generation
    -- Sub-class ImportExport
"""

from ImportAllS import *

class ImportExport():
    # Initialization
    def __init__(self,X=None):
        self.X = X
        return
    
    # Export to file
    def exportFile(self,format,filename='meshRef',build=False):
        """
        --- Creates a filename.format file and writes the results of A.generateNodes() on it
        A.exportFile(format=,filename=)
        or
        A.exportFile(format=)
        
        -- Input
        format          : str       format to export to
            - 'su2'
            - 'msh'
            - 'vtk'
        filename        : str       name of the  file (optionnal)              (default: 'meshRef')
        build           : boolean   generate the nodes again ? (optional)      (default: False)
        """
        if format == 'su2': self.exportToSU2(filename)
        elif format == 'msh': self.exportToMSH(filename)
        elif format == 'vtk': self.exportToVTK(filename)
        else: print("Error in format")
        return
            
    def exportToSU2(self,filename='meshRef'):
        """
        --- Creates a filename.su2 file and writes the results of A.generateNodes() on it
        A.exportToSU2()
        or
        A.exportToSU2(filename)
        
        -- Input
        filename        : str       name of the  file
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".su2"
        su2_file = open(filename,'w+')
        su2_file.write("% Mesh dimension\n")
        su2_file.write("NDIME= 2\n")
        su2_file.write("\n")
        
        # Nodes input
        su2_file.write("% Nodes coordinates\n")
        su2_file.write("NPOIN= {}\n".format(self.N))
        for i in range(0,self.N):
            su2_file.write("{X:.16f} {Y:.16f} {k}\n".format(X=self.X[i,0],Y=self.X[i,1],k=i))
        
        #Elements input
        su2_file.write("\n% Element list\n")
        su2_file.write("NELEM= {}\n".format(self.N_el))
        el = self.generateElements()
        for i in range(0,self.N_el):
            su2_file.write("4 {N1} {N2} {N3} {N4}\n".format(N1=el[i,0],N2=el[i,1],N3=el[i,2],N4=el[i,3]))
        
        #Limits (Markers)
        su2_file.write("\n% Markers list\n")
        su2_file.write("NMARK= 4\n")
        
        nMarker_lower = self.n - 1
        nMarker_side = self.m - 1
        
        su2_file.write("MARKER_TAG= lower\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_lower))
        for i in range(0,self.n-1):
            su2_file.write("{type} {N1} {N2}\n".format(type=3,N1=i,N2=i+1))
        
        su2_file.write("MARKER_TAG= upper\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_lower))
        for i in range(0,self.n-1):
            su2_file.write("3 {N1} {N2}\n".format(N1=(self.m-1)*self.n + i,N2=(self.m-1)*self.n + i+1))
        
        su2_file.write("MARKER_TAG= right\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_side))
        for i in range(0,self.m-1):
            su2_file.write("3 {N1} {N2}\n".format(N1=self.n -1 + (self.m+1)*i,N2=self.n -1 + (self.m+1)*(i+1)))
        
        su2_file.write("MARKER_TAG= left\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_side))
        for i in range(0,self.m-1):
            su2_file.write("3 {N1} {N2}\n".format(N1= + (self.m+1)*i,N2= + (self.m+1)*(i+1)))
        
        su2_file.close()
        return
        
    def exportToMSH(self,filename='meshRef'):
        """
        --- Creates a filename.msh file and writes the results of A.generateNodes() on it
        A.exportToMSH()
        or
        A.exportToMSH(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".msh"
        msh_file = open(filename,'w+')
        msh_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # Nodes input
        msh_file.write("$Nodes\n")
        msh_file.write("{}\n".format(self.N))
        for i in range(0,self.N):
            msh_file.write("{k} {X:.16f} {Y:.16f} {Z:.16f}\n".format(X=self.X[i,0],Y=self.X[i,1],k=i+1,Z=0))
        msh_file.write("$EndNodes\n")
        
        #Elements input
        msh_file.write("$Elements\n")
        msh_file.write("{}\n".format(self.N_el))
        el = self.generateElements()
        for i in range(0,self.N_el):
            msh_file.write("{K}  3  2  0  {K}  {N1}  {N2}  {N3}  {N4}\n".format(K=i+1,N1=el[i,0]+1,N2=el[i,1]+1,N3=el[i,2]+1,N4=el[i,3]+1))
        msh_file.write("$EndElements\n")
        msh_file.close()
        return

    def exportToVTK(self,filename='meshRef'):
        """
        --- Creates a filename.vtk file and writes the results of A.generateNodes() on it
        A.exportToVTK()
        or
        A.exportToVTK(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".vtk"
        vtk_file = open(filename,'w+')
        vtk_file.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        
        # Nodes (Points) input
        vtk_file.write("POINTS {} float\n".format(self.N))
        for i in range(0,self.N):
            vtk_file.write("{X:.16f} {Y:.16f} {Z:.16f}\n".format(X=self.X[i,0],Y=self.X[i,1],Z=0))
        vtk_file.write("\n")

        #Elements (Cells) input
        vtk_file.write("CELLS {N_el} {size} \n".format(N_el=self.N_el,size=self.N_el*5))
        el = self.generateElements()
        for i in range(0,self.N_el):
            vtk_file.write("4 {N1} {N2} {N3} {N4}\n".format(N1=el[i,0],N2=el[i,1],N3=el[i,2],N4=el[i,3]))
        vtk_file.write("\n")
        
        #Elements (Cell_Type) input
        vtk_file.write("CELL_TYPES {N_el}\n".format(N_el=self.N_el))
        el = self.generateElements()
        for i in range(0,self.N_el):
            vtk_file.write("9\n")
        vtk_file.close()
        return
        
    
    # Import from file
    def importFile(self,format,filename='meshRef'):
        #############################################################################################
        # OBSOLETE
        # IMPORTANT TO NOTE THAT THIS ONLY GIVES n AND m SO IT ACTUALLY DOESN'T WORK ON PHYSICAL MESH
        # AND DOESN'T SEND BACK THE POINTS X, elem, ETC.
        #############################################################################################
        """
        --- Creates a filename.format file and send the results of self.n and self.m
        A.importFile(format=,filename=)
        or
        A.importFile(format=)
        
        -- Input
        format          : str       format of the the file to import from
            - 'su2'
            - 'msh'
            - 'vtk'
        filename        : str       name of the  file (optionnal)
        """
        if format == 'su2': self.importFromSU2(filename)
        if format == 'msh': self.importFromMSH(filename)
        if format == 'vtk': self.importFromVTK(filename)
        return self.n, self.m
    
    def importFromSU2(self,filename='meshRef'):
        """
        --- Reads a filename.su2 file and sends back the results
        A.importFromSU2()
        or
        A.importFromSU2(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.su2'
        su2_file = open(filename,'r')
        file = su2_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')

        # Search for "MARKER_TAG= lower" : Extracts n
        for i in range(0,len(file)):
            if file[i].find("MARKER_TAG= lower") == 0:
                self.n = int(file[i+1].strip("MARKER_ELEMS= ")) + 1
                break
        
        # Search for "MARKER_TAG= right" : Extacts m
        for i in range(0,len(file)):
            if file[i].find("MARKER_TAG= right") == 0:
                self.m = int(file[i+1].strip("MARKER_ELEMS= ")) + 1
                break
        su2_file.close()
        return
        
    def importFromMSH(self,filename='meshRef'):
        """
        --- Reads a filename.msh file and writes the results of A.generateNodes() in self.X_ref
        A.importFromMSH()
        or
        A.importFromMSH(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.msh'
        msh_file = open(filename,'r')
        file = msh_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')

        # Search for "$Nodes" : Extracts N
        for i in range(0,len(file)):
            if file[i].find("$Nodes") == 0:
                pos = i
                N = int(file[i+1])
                break
        
        # Search for "$Elements" : Extracts N_el
        for i in range(0,len(file)):
            if file[i].find("$Elements") == 0:
                N_el = int(file[i+1])
                break
        
        # Extract X_ref to determine n
        X_ref = np.zeros((N,2))
        for i in range(pos+2,pos+2+N):
            X_ref[i-5,0] = file[i][len(str(i)):len(str(i))+16]
            X_ref[i-5,1] = file[i][len(str(i))+19:len(str(i))+19*2]

        # Extraction of n            
        for i in range(0,N):
            self.n = i+1
            if X_ref[i,0] == 1: break
        
        # Computing of m : m = N/n
        self.m = int(N/self.n)
        msh_file.close()
        return
        
    def importFromVTK(self,filename='meshRef'):
        """
        --- Reads a filename.vtk file and writes the results of A.generateNodes() in self.X_ref
        A.importFromVTK()
        or
        A.importFromVTK(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.vtk'
        vtk_file = open(filename,'r')
        file = vtk_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')
        
        # Search for "POINTS " : Extracts N
        pos = 3
        N = int(file[4][7:len(file[4])].strip('float'))
        
        # Search for "$Elements" : Extracts N_el
        for i in range(0,len(file)):
            if file[i].find("$Elements") == 0:
                N_el = int(file[i+1])
                break
        
        # Extract X_ref to determine n
        X_ref = np.zeros((N,2))
        for i in range(pos+2,pos+2+N):
            X_ref[i-5,0] = file[i][0:16]
            X_ref[i-5,1] = file[i][19:19*2]

        # Extraction of n            
        for i in range(0,N):
            self.n = i+1
            if X_ref[i,0] == 1: break

        # Computing of m : m = N/n
        self.m = int(N/self.n)
        vtk_file.close()
        return