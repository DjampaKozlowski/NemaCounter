import os
import psutil
import torch

def list_image_files(dir_path):
    """
    List images with the '.jpg', '.jpeg', '.png' extensions
    in a given directory or subdirectories
    """
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more extensions if needed
    image_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files

def check_file_existence(path):
    """
    Check is a path exists and if it points to a file (and not a dir)
    """
    return os.path.isfile(path)

def get_bool(arg):
    if arg==1:
        return True
    else:
        return False

def compute_available_cpu():
    """
    returns rounding to the lower integer
    """
    return int(psutil.cpu_count()*(1-psutil.cpu_percent()/100))

def set_cpu_usage(n):
    """
    Set the number of CPU to be used by pytorch. 
    
    If -1 (default value), all the available cpus can be 
    used by torch. Else, use the given number of CPU max.
    """
    if n != -1:
        avail_cpu = compute_available_cpu()
        if n <= avail_cpu:
            torch.set_num_threads(n)
        else:
            torch.set_num_threads(avail_cpu)
    else:
        torch.set_num_threads(avail_cpu)
        
        







