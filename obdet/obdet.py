import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image


#################################################
# Pixels Calculations
#################################################

def cal_mode(x):
    """
    Calculate the mode value from given array
    :param x: image in form of array
    :return mode of the numpy array:
    """
    x = x.flatten()
    idx, cnts = np.unique(x, return_counts=True)
    nid = np.argmax(cnts)
    return idx[nid]


def get_img_path(home_path, f_name):
    """
    Extract the path of image from .txt file
    :param home_path: directory for dataset
    :param f_name: file name
    :return: image path
    """
    fn_list = f_name.split('.')
    img_name, ext = '.'.join(fn_list[:-1]), fn_list[-1]
    for extension in ['.jpg', '.png']:
        img_path = os.path.join(home_path, img_name + extension)
        if os.path.isfile(img_path):
            return img_path


def get_current_image(img_path, img_size):
    """
    Open a image from given filepath
    :param img_path:
    :param img_size:
    :return:
    """
    pil_img = Image.open(img_path)
    img_height, img_width = pil_img.size
    if img_size and (img_height != img_width):
        print('\nsize parameters must be defined carefully.')
        print('your images are not square\n')
        exit(1)
    return pil_img


def get_img_vals(pil_img=None):
    """
    Take the image and calculate mode value for every different channel
    :param pil_img:
    :return:
    """
    np_img = np.asarray(pil_img)

    raf = np_img[..., 0]
    gaf = np_img[..., 1]
    baf = np_img[..., 2]

    r_mode = cal_mode(raf)
    g_mode = cal_mode(gaf)
    b_mode = cal_mode(baf)

    return r_mode, g_mode, b_mode


def get_obj_coordinates(curr_image, obj_data):
    """
    Extract the cropped version of the object from the given image
    :param curr_image:
    :param obj_data:
    :return:
    """
    c_img_w = curr_image.size[0]
    c_img_h = curr_image.size[1]

    cx = (float(obj_data[1]) * c_img_w)
    cy = (float(obj_data[2]) * c_img_h)
    bw = (float(obj_data[3]) * c_img_w)
    bh = (float(obj_data[4]) * c_img_h)
    # print(obj_data)

    left = int(cx - (bw / 2))
    top = int(cy - (bh / 2))
    right = int(cx + (bw / 2))
    bottom = int(cy + (bh / 2))

    cropped_img = curr_image.crop((left, top, right, bottom))
    return get_img_vals(cropped_img)


def extract_color_info(class_grp):
    """
    Create a dictionary of color values against each class
    :param class_grp:
    :return:
    """
    data_dict = dict()
    for class_item in tqdm(class_grp, desc='[INFO]: Extracting Object Information:'):
        class_name = class_item[0]
        class_df = class_item[1]
        class_np = np.zeros((class_df.shape[0], 4, 4, 3))
        # print(class_df.shape[0])
        for idx, (r_idx, class_df_row) in enumerate(class_df.iterrows()):
            r_val = class_df_row['obj_r_mode']
            g_val = class_df_row['obj_g_mode']
            b_val = class_df_row['obj_b_mode']

            class_np[idx][..., 0].fill(r_val)
            class_np[idx][..., 1].fill(g_val)
            class_np[idx][..., 2].fill(b_val)

        data_dict.update({class_name: class_np.astype('uint8')})
    return data_dict


def save_obj_images(images, class_name='', bg='white', rd=''):
    """
    Create the color charts for the objects
    :param images:
    :param class_name:
    :param bg:
    :param rd:
    :return:
    """
    plt.clf()
    images_per_row = 16

    n_features = images.shape[0]
    size = images.shape[1]
    n_cols = n_features // images_per_row

    # print(n_features, size, n_cols)
    # print((size + 1)* n_cols - 1,)

    grid_start = (size + 1) * n_cols - 1
    grid_end = images_per_row * (size + 1) - 1
    if bg.lower() == 'white' or bg.lower() == 'w':
        display_grid = np.ones((grid_start, grid_end, 3))
    else:
        display_grid = np.zeros((grid_start, grid_end, 3))

    # print(display_grid.shape)
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = images[channel_index, ...].copy()

            start_col = col * (size + 1)
            end_col = (col + 1) * size + col
            start_row = row * (size + 1)
            end_row = (row + 1) * size + row

            display_grid[start_col: end_col, start_row: end_row, :] = channel_image / 255.

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto")
    plt.savefig(f'./results/{rd}/color_chart_{class_name}.jpg')


#################################################
# Analytics Functions
#################################################


def check_details(details_level=''):
    return details_level.lower() == 'all'


def print_len(home_path):
    """
    This function print and create the list of all available files in YOLO format
    :param home_path:
    :return file_list:
    """
    files_list = os.listdir(home_path)
    files_list.sort()
    print(f'Total Image Files: {int(len(files_list) / 2)}')
    return files_list


def get_columns_list():
    return ['img_path',
            'img_size',
            'img_width',
            'img_height',

            'class_name',
            'class_idx',

            'ncx',
            'ncy',
            'nbw',
            'nbh']


def create_dataframe(home_path,
                     img_size, img_h, img_w,
                     files_list,
                     is_all=False):
    """
    Create the dataframe containing all the data related to image and labels
    :param home_path:
    :param img_size:
    :param img_h:
    :param img_w:
    :param files_list:
    :param is_all:
    :return:
    """
    if img_size:
        image_size = img_size * img_size
        image_height, image_width = img_size, img_size
    else:
        image_size = img_h * img_w
        image_height, image_width = img_h, img_w

    total_list = []

    class_file = os.path.join(home_path, 'classes.txt')
    with open(class_file) as fc:
        class_list = fc.readlines()

    print('[INFO]: Extracting information from the dataset...')
    for file_name in tqdm(files_list):
        if file_name.endswith('.txt') and not file_name.startswith('class'):
            file_path = os.path.join(home_path, file_name)

            if is_all:
                current_image = get_current_image((get_img_path(home_path, file_name)), img_size)
                im_r_mode, im_g_mode, im_b_mode = get_img_vals(current_image)

            with open(file_path) as fp:

                file_data = fp.readlines()

                for data_list in file_data:
                    data_item = data_list.split(' ')
                    if is_all:
                        obj_r_mode, obj_g_mode, obj_b_mode = get_obj_coordinates(current_image, data_item)
                    try:
                        item_list = [
                            file_path,

                            image_size,
                            image_width,
                            image_height,

                            class_list[int(data_item[0])].replace('\n', ''),
                            int(data_item[0]),

                            float(data_item[1]),
                            float(data_item[2]),
                            float(data_item[3]),
                            float(data_item[4]),

                        ]

                        if is_all:
                            item_list.extend([
                                int(im_r_mode),
                                int(im_g_mode),
                                int(im_b_mode),

                                int(obj_r_mode),
                                int(obj_g_mode),
                                int(obj_b_mode),
                            ])

                        total_list.append(item_list)

                    except IndexError:
                        print('\nClass names must match with # of objects')
                        print('Please check your object labels or total of class names.\n')
                        exit(1)

    columns_list = get_columns_list()
    if is_all:
        columns_list.extend(['im_r_mode', 'im_g_mode', 'im_b_mode',
                             'obj_r_mode', 'obj_g_mode', 'obj_b_mode', ])

    df = pd.DataFrame(data=total_list,
                      columns=columns_list)

    df['cx'], df['cy'], df['bw'], df[
        'bh'] = df.ncx * image_width, df.ncy * image_height, df.nbw * image_width, df.nbh * image_height
    df['obj_px'] = df['bw'] * df['bh']
    df['obj_img_ratio'] = df['obj_px'] / df['img_size']
    print(f'Total Objects in the dataset: {df.shape[0]}\n')
    return df


def draw_pie_chart(dfx, rd):
    """
    Generate pie chart from the grouped dataframe
    :param dfx:
    :param rd:
    :return:
    """
    print('[INFO]: Drawing pie chart for classes...')
    class_distribution = dfx.count()['img_size']
    plt.pie(class_distribution,
            labels=[f'{k}: {v}' for k, v in class_distribution.items()]
            , autopct='%1.1f%%')
    plt.legend(title='Class Distribution')
    plt.savefig(f'./results/{rd}/class_distribution.png')
    plt.clf()


def draw_bar_chart(dfx, rd):
    """
    Generate horizontal bar charts against your object detection dataset
    :param dfx:
    :param rd:
    :return:
    """
    # Mean BB Pixel
    print('[INFO]: Drawing bar chart for mean pixel distribution...')
    dfx.mean()['obj_px'].plot(kind='barh')
    plt.ylabel('pixel size')
    plt.title('Mean Pixel Size BBox')
    plt.legend()
    plt.savefig(f'./results/{rd}/mean_bbpixel_size.png')
    plt.clf()


def draw_class_dist(dfx, rd):
    """
    Visualize the distribution of pixels among all the classes
    :param dfx:
    :param rd:
    :return:
    """
    print('[INFO]: Drawing classes bbox sizes...')
    ymax_limit = dfx.count().max()[0]
    for class_name, class_df in dfx:
        class_df['obj_px'].plot(kind='hist', bins=5, rwidth=0.85)
        plt.ylim(ymax=ymax_limit)
        plt.xlabel('BB Size in pixels')
        plt.title(f"Class: '{class_name}' Pixel Variation")
        plt.savefig(f'./results/{rd}/bb_px_dist_{class_name}.png')
        plt.clf()


def draw_color_chart(dfx, rd):
    """
    Generate the color chart based on object level info
    :param dfx:
    :param rd:
    :return:
    """
    color_data = extract_color_info(dfx)
    # print('Creating object colors charts...')
    t = tqdm(color_data.items(), '[INFO]: Starting')
    for c_name, c_images in t:
        t.set_description(f'[INFO]: Creating charts for {c_name}')
        save_obj_images(c_images, c_name, bg='w', rd=rd)


def generate_analytics(home_path='./dataset/ts',
                       img_size=None, img_h=None, img_w=None,
                       details_level='dist'):
    """
    Generate object detection analytics for the given directory
    :param home_path: Path of the directory containing dataset
    :param img_size: Images Size (if square)
    :param img_h: Images Height (if non-square)
    :param img_w: Images Width (if non-square)
    :param details_level: Level of details for analysis
    :return:
    """
    print('[INFO]: Pipeline Starting...\n')
    f_list = print_len(home_path)

    is_all = check_details(details_level)

    df = create_dataframe(home_path=home_path,
                          img_size=img_size, img_h=img_h, img_w=img_w,
                          files_list=f_list,
                          is_all=is_all)

    df_class_grp = df.groupby(['class_name'])
    os.makedirs('./results', exist_ok=True)
    res_dir_name = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    os.makedirs(f'./results/{res_dir_name}', exist_ok=True)

    draw_pie_chart(df_class_grp, res_dir_name)
    draw_bar_chart(df_class_grp, res_dir_name)
    draw_class_dist(df_class_grp, res_dir_name)
    if is_all:
        draw_color_chart(df_class_grp, res_dir_name)

    print('\n[SUCCESS]: Analytics generated successfully, please check the results folder...\n')
