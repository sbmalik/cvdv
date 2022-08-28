import argparse
from datetime import datetime
import os
import re
from PIL import Image
import pandas as pd

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
    print('[INFO]: Data directory is a valid path...\n')
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

def get_current_image(img_path):
    """
    Open a image from given filepath
    :param img_path: filepath of the image
    :return pil_img: PIL image object
    """
    if os.path.isdir(img_path):
        raise Exception("{} is not a valid image".format(img_path))
    pil_img = Image.open(img_path)
    return pil_img


def create_dataframe(classes_data):
    """
    Create a dataframe for the analysis of extracted data

    Args:
        classes_data (dict): data containing the classes and their data

    Returns:
        df (DataFrame): Dataframe containing the required insights
    """
    print('[INFO]: Creating dataframe for extracted data...')
    total_list = []
    for cls, img_path_list in classes_data.items():
        for img_path in img_path_list:
            im_w, im_h = get_current_image(img_path).size
            total_list.append([
                cls,
                img_path,
                im_w,
                im_h
            ])
    df = pd.DataFrame(total_list, columns=[
                      'class_name', 'img_path', 'img_w', 'img_h'])

    print(f'[INFO]: Total Objects in the dataset: {df.shape[0]}\n')
    return df


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
                classes_data[cls] = tuple(
                    os.path.join(home_path, cls, img_path)
                    for img_path in imgs_list
                )

    print('[INFO]: Total Classes: {}'.format(len(classes_data)))
    print('[INFO]: Classes: {}'.format(", ".join(classes_data.keys())))
    return classes_data


def generate_analytics(home_path=''):
    print('[INFO]: Pipeline Starting...\n')
    cls_data = get_classes_data(home_path)
    df = create_dataframe(cls_data)
    df_class_grp = df.groupby(['class_name'])
    os.makedirs('./results', exist_ok=True)
    res_dir_name = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    os.makedirs(f'./results/{res_dir_name}', exist_ok=True)
    df.to_csv(f'./results/{res_dir_name}/analysis.csv', index=False)
    print(f"\n[SUCCESS]: Analytics generated successfully,"
          f' please check the results folder...\n')


def perform_analysis(var_args):
    """
    Perform the analysis on the given dataset

    Args:
        var_args (dict): Dictionary of arguments
    """
    generate_analytics(home_path=var_args.data_dir)
