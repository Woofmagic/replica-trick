# Native Library | datetime
import datetime

from app.utilities.directories.handling_directories import create_replica_directories
from app.utilities.directories.handling_directories import create_replica_model_directories
from app.utilities.directories.handling_directories import create_replica_plots_directories
from app.utilities.directories.handling_directories import find_replica_model_directories
from app.utilities.directories.handling_directories import find_replica_plots_directories

def run_replica_method(
        kinematic_set_integer: int = 1,
        number_of_replicas: int = 1,
        verbose: bool = False):
    
    # (1): Begin iterating over the replicas:
    for replica_index in range(number_of_replicas):

        # (1.1): Obtain the replica number by adding 1 to the index:
        replica_number = replica_index + 1

        # (1.2): Propose a replica name:
        current_replica_name = f"replica_{replica_number}"

        # (1.3): Immediately construct the filetype for the replica:
        model_file_name = f"{current_replica_name}.h5"

        # (1.4): Create the directory for the replica:
        did_we_create_replica_directory = create_replica_directories(kinematic_set_integer, replica_number)
        did_we_create_replica_model_directory = create_replica_model_directories(kinematic_set_integer, replica_number)
        did_we_create_replica_plots_directory = create_replica_plots_directories(kinematic_set_integer, replica_number)

        # (1.5): Begin timing the replica time:
        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        if verbose:
            print(f"> Replica #{replica_index + 1} now running...")