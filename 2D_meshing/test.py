import os


path = os.getcwd()
print(path)
path = path.replace("\\",'/')
print(path)
os.chdir(path)

path = os.getcwd()
print(path)