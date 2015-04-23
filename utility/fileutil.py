import os


# if directory not exist, create it
def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


# get parent dir of given path
def parent_dir_of(path):
    ppath = os.path.realpath(os.path.join(path, '..'))
    if os.path.isdir(path):
        return ppath
    elif os.path.isfile(path):
        return parent_dir_of(ppath)
    else:
        print str.format('ERROR! This path: {0} is not exist.', path)
        return -1


# get num at (row, col) in a file
def num_in_file(filepath, row, col, dtype=int):
    f = open(filepath, 'r')
    lines = f.readlines()
    return dtype(lines[row-1].strip().split()[col-1])