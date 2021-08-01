import logging

import pandas as pd
import matplotlib.pyplot as plt
import os
import traceback


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


def create_dataframe(home_path,
                     img_size,
                     files_list):
    """
    Create the dataframe containing all the data related to image and labels
    :param home_path:
    :param img_size:
    :param files_list:
    :return:
    """
    total_list = []

    class_file = os.path.join(home_path, 'classes.txt')
    with open(class_file) as fc:
        class_list = fc.readlines()

    for file_name in files_list:
        if file_name.endswith('.txt') and not file_name.startswith('class'):
            file_path = os.path.join(home_path, file_name)
            with open(file_path) as fp:
                # print(file_name)
                file_data = fp.readlines()
                for data_list in file_data:
                    data_item = data_list.split(' ')
                    try:
                        total_list.append([
                            file_path,
                            img_size * img_size,
                            img_size,
                            img_size,
                            class_list[int(data_item[0])].replace('\n', ''),
                            int(data_item[0]),
                            float(data_item[1]),
                            float(data_item[2]),
                            float(data_item[3]),
                            float(data_item[4])
                        ])
                    except IndexError:
                        print('\nClass names must match with # of objects')
                        print('Please check your object labels or total of class names.\n')
                        exit(0)

    df = pd.DataFrame(data=total_list,
                      columns=['img_path',
                               'img_size',
                               'img_width',
                               'img_height',
                               'class_name',
                               'class_idx',
                               'ncx',
                               'ncy',
                               'nbw',
                               'nbh'])

    df['cx'], df['cy'], df['bw'], df[
        'bh'] = df.ncx * img_size, df.ncy * img_size, df.nbw * img_size, df.nbh * img_size
    df['obj_px'] = df['bw'] * df['bh']
    df['obj_img_ratio'] = df['obj_px'] / df['img_size']
    print(f'Total Objects in the dataset: {df.shape[0]}')
    return df


def draw_pie_chart(dfx):
    """
    Generate pie chart from the grouped dataframe
    :param dfx:
    :return:
    """
    class_distribution = dfx.count()['img_size']
    plt.pie(class_distribution,
            labels=[f'{k}: {v}' for k, v in class_distribution.items()]
            , autopct='%1.1f%%')
    plt.legend(title='Class Distribution')
    plt.savefig('./results/class_distribution.png')
    plt.clf()


def draw_bar_chart(dfx):
    """
    Generate horizontal bar charts against your object detection dataset
    :param dfx:
    :return:
    """
    # Mean BB Pixel
    dfx.mean()['obj_px'].plot(kind='barh')
    plt.ylabel('pixel size')
    plt.title('Mean Pixel Size BB')
    plt.legend()
    plt.savefig('./results/mean_bbpixel_size.png')
    plt.clf()


def draw_class_dist(dfx):
    """
    Visualize the distribution of pixels among all the classes
    :param dfx:
    :return:
    """
    for class_name, class_df in dfx:
        class_df['obj_px'].plot(kind='hist', bins=5, rwidth=0.85)
        plt.ylim(ymax=1000)
        plt.xlabel('BB Size in pixels')
        plt.title(f"Class: '{class_name}' Pixel Variation")
        plt.savefig(f'./results/bb_px_dist_{class_name}.png')
        plt.clf()


def generate_analytics(home_path='./dataset/ts',
                       img_size=544):
    """
    Generate object detection analytics for the given directory
    :param home_path:
    :param img_size:
    :return:
    """
    f_list = print_len(home_path)
    df = create_dataframe(home_path=home_path,
                          img_size=img_size,
                          files_list=f_list)
    df_class_grp = df.groupby(['class_name'])
    os.makedirs('./results', exist_ok=True)
    draw_pie_chart(df_class_grp)
    draw_bar_chart(df_class_grp)
    draw_class_dist(df_class_grp)
    print('Analytics generated successfully, please check the results folder...')
