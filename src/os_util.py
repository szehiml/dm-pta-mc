"""
    Collection of useful functions for saving/moving/creating files/directories    
"""
import os

def my_mkdir(dir_name):
    
    # make a directory with name dir_name if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
