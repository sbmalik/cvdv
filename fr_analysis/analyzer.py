import argparse
import os
import re

# python3 main.py --data_dir /home/darvis-ml-3/darvis_ml/xperiments/dst_six_1/services/deepstream/exp/dst_python_apps/data/embeddings_root4/

#################################################
# Require Parameters
#################################################

def check_params(args):
    """
    Check if the parameters are valid
    """
    if not os.path.isdir(args.data_dir):
        raise Exception("{} is not a valid directory".format(args.data_dir))
    print('[INFO]: Data directory is a valid path...')
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
    return args


#################################################
# Analytics Functions
#################################################

def get_classes_data(home_path=''):
    """
    Get the classes and their data from the dataset

    Args:
        home_path (str, optional): Data directory. Defaults to ''.

    Returns:
        classes_data (dict): data containing the classes and their data
    """
    print('[INFO]: Getting classes from the dataset...')
    classes = os.listdir(home_path)
    classes_data = {}
    for cls in classes:
        if cls.startswith('.'):
            continue 
        else:
            cls_imgs = os.listdir(os.path.join(home_path, cls))
            r = re.compile(".*jpg")
            imgs_list = list(filter(r.match, cls_imgs))
            if len(imgs_list) == 0:
                continue
            else:
                classes_data[cls] = imgs_list

    print('[INFO]: Total Classes: {}'.format(len(classes_data)))
    print('[INFO]: Classes: {}'.format(", ".join(classes_data.keys())))
    return classes_data

def generate_analytics(home_path=''):
    print('[INFO]: Pipeline Starting...\n')
    cls_dsts = get_classes_data(home_path)

def perform_analysis(var_args):
    """
    Perform the analysis on the given dataset

    Args:
        var_args (dict): Dictionary of arguments
    """
    generate_analytics(home_path=var_args.data_dir)