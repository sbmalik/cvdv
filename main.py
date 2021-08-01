from obdet import obdet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        help='PATH of the dataset',
                        required=True)
    parser.add_argument("--im_size",
                        type=int,
                        help='Single dimension of the dataset image [MUST be Squared]',
                        required=True)
    args = parser.parse_args()
    HOME_PATH = args.data_dir
    IMAGE_SIZE = args.im_size
    obdet.generate_analytics(home_path=HOME_PATH,
                             img_size=IMAGE_SIZE)
