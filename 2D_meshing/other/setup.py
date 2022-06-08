"""
Used to setup the notebook with the programm << toolsRef.py >>
Use in a python file or notebook:
    from setup import *
    
Working directory needs to be at least ".../2D_meshing"
"""
import os

def mainPath():
    path_main = os.getcwd()
    # Generalize the path to work on both Windows and Linux
    path_main = path_main.replace("\\",'/')

    # Locate the "/2D_meshing" in the path to set the main path the 2D_meshing tools
    main = path_main.find('/2D_meshing')
    if main == -1:
        print("Current working directory needs to be at least in .../2D_meshing !")
    main += 11
    return path_main[:main]

path_main = mainPath()

path = path_main + "/src"


path_main = __file__.replace("\\","/")
path_main = path_main.replace("/notebook/setup.py","")
path_main = path_main + "/src"
os.chdir(path_main)