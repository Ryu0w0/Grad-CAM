import os


def create_folder(full_path):
    """ Make directory if specified patch does not exist """
    dir_list = full_path.split("/")
    for i in range(len(dir_list)):
        chk_path = "/".join(dir_list[0:i+1])
        if not os.path.exists(chk_path):
            os.mkdir(chk_path)