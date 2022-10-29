## not used...
import argparse
import dataclasses

# creates an empty configuration file for connection to a server
# need to handle saving/loading as a json file.
@dataclasses.dataclass
class RemoteServerConfiguration:

    server_name: str = "ds-serv1"  # make sure that ssh `server_name` works directly

    # the local scripts will be completely copied to the remote directory.
    # things here should be small in size, can be updated constantly
    # TODO what happens if remote is not empty. A check should be done that finds the replaced files, and extra files in remote.
    local_source_folder: str = './test_experiment/src'  # the local directory that will be completely copied to remote
    remote_source_folder: str = '/data1/zihan/test_experiment/src'  # the remote directory that will be replaced by local.

    # currently, we do no data syncing! so the `local_data_folder` is actually not used.
    # things here should be large in size, and only readable!
    # TODO handle data diffs, and if there is an update from the local directory, push that update to
    # TODO what if one needs to dynamically create a dataset, how could they be stored? This is not supported
    local_data_folder: str = './test_data'  # the local directory that contains the data
    remote_data_folder: str = '/data1/zihan/test_experiment/data'  # the remote directory that will hold the data

    # the remote directory that experiments happen, and output/model is stored
    # things there could be large/small. Need to specify what things should be pulled to local.
    remote_experiment_folder: str = '/data1/zihan/test_experiment/experiments' # the remote directory to store the run outputs


    local_remote_logs_folder: str = './test_experiment/remote_logs' # the folder to store connection logs to remote directory, will be prefixed by server_name.


# currently no file syncing, just returning the default thingy
# ideally project_configuration should be stored in local_source_folder, under a name like ds-serv1.config
def get_server_configuration(path_to_project_configuration):
    return RemoteServerConfiguration()
