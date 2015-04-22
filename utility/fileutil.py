import os


# if directory not exist, create it
def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
