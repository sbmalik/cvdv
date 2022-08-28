import argparse
import os

# 

#################################################
# Require Parameters
#################################################

def check_params(args):
    """
    Check if the parameters are valid
    """
    if not os.path.isdir(args.data_dir):
        raise Exception("{} is not a valid directory".format(args.data_dir))
    return args

def get_params():
    """
    Get parameters from command line for object detection analysis

    Returns:
        args: variable containing all the parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help='PATH of the dataset',
                        required=True)
    args = parser.parse_args()
    check_params(args)