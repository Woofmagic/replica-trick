# Native Library | os
import os

# Native Library | errno
import errno

# Native Library | pathlib
from pathlib import Path

# statics > strings > directory names
from app.statics.strings.static_strings import _DIRECTORY_DATA
from app.statics.strings.static_strings import _DIRECTORY_EXTRACTIONS_MODELS_
from app.statics.strings.static_strings import _DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS

def find_directory(base_directory: str, additional_directory_info: str) -> os.path:
    """
    # Description:

    Look for a directory by stitching together a base_directory
    (usually the pwd) and the "hypothesized" path where the file
    you're looking for lives.

    # Arguments:

    base_directory: str
    additional_directory_info: str
    
    # Returns

    directory_path: bool | False
    """
    try:
        directory_path = os.path.join(base_directory, additional_directory_info)
        return directory_path
    except Exception as ERROR:
        print(f"> Error in finding directory:\n{ERROR}")
        
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), f"{base_directory}/{additional_directory_info}")
    
def does_directory_exist(path_to_directory: os.path) -> bool:
    """
    # Description:

    Here, we just look for a given os.path within a given
    directory context.

    # Arguments:

    path_to_directory: os.path

    # Returns:

    does_the_path_exist: bool | False
    """

    try:

        does_the_path_exist = os.path.exists(path_to_directory)
        return does_the_path_exist
    
    except Exception as ERROR:
        print(f"> Error in finding directory existence:\n{ERROR}")
        return False
    
def create_directory(filepath_to_directory):
    """
    # Title: `create_directory`

    ## Description: 
    
    ## Arguments:

        1. filepath_to_directory (str)

    ## Returns:
    """

    if does_directory_exist(filepath_to_directory):

        return False
    
    else:

        try:

            did_os_make_the_directory = os.makedirs(filepath_to_directory)
            return did_os_make_the_directory
        
        except Exception as ERROR:

            print(f"> Error in creating the directory:\n{ERROR}")
            return False

def create_kinematic_set_directories(kinematic_set_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/`
    kinematic_sets_filepath = f"{current_working_directory}//{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}"

    # (3): We create the filepath `root/data/models/kinematic_sets/set_number`
    did_we_create_kinematic_set_directory = create_directory(f"{kinematic_sets_filepath}//kinematic_set_{kinematic_set_number}")

    return did_we_create_kinematic_set_directory

def create_replica_directories(kinematic_set_number, replica_number):
    """
    # Title: `create_replica_directories`

    ## Description:

    ## Arguments:

        1. kinematic_set_number (int)
        2. replica_number (int)
    """

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N`
    kinematic_set_n_filepath = f"{current_working_directory}//{_DIRECTORY_DATA}//analysis//replicas//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}"

    try:

            # (3): We create the filepath `root/data/models/kinematic_sets/kinematic_set_N/replica_X`
        did_we_create_replica_directory = create_directory(f"{kinematic_set_n_filepath}//replica_{replica_number}")

        return did_we_create_replica_directory
    
    except Exception as ERROR:
        print(f"> Error in creating directory! {ERROR}")
        return False

def find_replica_directories(kinematic_set_number: int):
    """
    # Description:

    We are looking for the directory with all the replicas. For example,
    it is looking for the following (static) folders: `replica_1/`, `replica_2/`...

    base/
        |-- data/
            |-- models/
                |-- kinematic_sets/
                    |-- replica_1/
                    
                    |-- replica_2/
                    ...

    # Arguments:

    kinematic_set_number: int

    # Returns:

    directory_with_replicas: path??
    """
    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N`
    kinematic_set_n_filepath = f"{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}"

    # (3): We create the filepath `root/data/models/kinematic_sets/kinematic_set_N/replica_X`
    directory_with_replicas = find_directory(current_working_directory, kinematic_set_n_filepath)

    return directory_with_replicas

def create_replica_model_directories(kinematic_set_number, replica_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N/replica_X`
    replica_X_filepath = f"{current_working_directory}//{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}//replica_{replica_number}"

    did_we_create_replica_model_directory = create_directory(f"{replica_X_filepath}//model")

    return did_we_create_replica_model_directory

def find_replica_model_directories(kinematic_set_number, replica_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N/replica_X/model/`
    replica_X_model_Y_filepath = f"{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}//replica_{replica_number}//model"

    did_we_create_replica_model_directory = find_directory(current_working_directory, replica_X_model_Y_filepath)

    return did_we_create_replica_model_directory

def create_replica_plots_directories(kinematic_set_number, replica_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N/replica_X`
    replica_X_filepath = f"{current_working_directory}//{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}//replica_{replica_number}"

    did_we_create_replica_model_directory = create_directory(f"{replica_X_filepath}//plots")

    return did_we_create_replica_model_directory

def find_replica_plots_directories(kinematic_set_number, replica_number):

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the relevant filepath: `data/models/kinematic_sets/kinematic_set_N/replica_X/plots/`
    replica_X_plots_filepath = f"{_DIRECTORY_DATA}//{_DIRECTORY_EXTRACTIONS_MODELS_}//{_DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS}//kinematic_set_{kinematic_set_number}//replica_{replica_number}//plots"

    did_we_create_replica_plots_directory = find_directory(current_working_directory, replica_X_plots_filepath)

    return did_we_create_replica_plots_directory

def find_all_model_paths(kinematic_set_number):
    """
    """

    # (1): Get the current working directory where `main.py` is running in:
    current_working_directory = os.getcwd()

    # (2): Construct the base path:
    base_path = Path(current_working_directory) / _DIRECTORY_DATA / _DIRECTORY_EXTRACTIONS_MODELS_ / _DIRECTORY_EXTRACTIONS_MODELS_KINEMATIC_SETS / f"{kinematic_set_number}"

    print(base_path)
    print(base_path.glob('replica_*/model/*.keras'))

    model_paths = []

    for replica_directory in base_path.glob('replica_*/model/*.keras'):
        model_paths.append(replica_directory)

    return model_paths