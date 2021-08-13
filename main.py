from obdet import obdet
import argparse


def check_flags():
    if IMAGE_SIZE is None:
        if IMAGES_WIDTH is None or IMAGES_HEIGHT is None:
            print('\nsize parameters must be defined carefully.')
            print('define --im_size in case of square image or define both --img_h & --img_w.\n')
            exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help='PATH of the dataset',
                        required=True)
    parser.add_argument("--details_level",
                        help='Details level, dist:distribution or all:objects/images/dist',
                        default='dist')
    parser.add_argument("--im_size",
                        type=int,
                        help='Single dimension of the dataset image [MUST be Squared]',
                        default=None)
    parser.add_argument("--im_h",
                        type=int,
                        help='Height of the images',
                        default=None)
    parser.add_argument("--im_w",
                        type=int,
                        help='Width of the images',
                        default=None)
    args = parser.parse_args()

    HOME_PATH = args.data_dir
    IMAGE_SIZE = args.im_size
    DETAILS_LEVEL = args.details_level
    IMAGES_HEIGHT = args.im_h
    IMAGES_WIDTH = args.im_h

    check_flags()

    obdet.generate_analytics(home_path=HOME_PATH,
                             img_size=IMAGE_SIZE, img_h=IMAGES_HEIGHT, img_w=IMAGES_WIDTH,
                             details_level=DETAILS_LEVEL)
