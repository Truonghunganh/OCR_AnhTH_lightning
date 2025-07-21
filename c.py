import os

def xoa_file_theo_duoi(path,duoi):
    path=str(path)
    if os.path.isfile(path):
        if path.endswith(duoi): 
            os.remove(path)
    elif os.path.isdir(path):
        for i in os.listdir(path):
            path_i=os.path.join(path,i)
            xoa_file_theo_duoi(path_i)