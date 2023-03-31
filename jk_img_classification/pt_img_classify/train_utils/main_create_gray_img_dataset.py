import argparse
import os
import shutil
import cv2

from pt_img_classify.train_utils.common import get_3d_gray_img

color_imgs_dir = '../data/test_data'
gray_imgs_dir = '../data/test_data_gray'


def main(args):
    create_gray_dirs(args)
    write_gray_imgs(args)


def write_gray_imgs(args):
    for root, sub_dir, f_list in os.walk(args.color_imgs_dir):
        for f in f_list:
            src_file = os.path.join(root, f)
            out_dir = root.replace(args.color_imgs_dir, args.gray_imgs_dir)
            out_img_path = os.path.join(out_dir, f)
            out_img = get_3d_gray_img(src_file)
            cv2.imwrite(out_img_path, out_img)


def create_gray_dirs(args):
    for root, sub_dir, f_list in os.walk(args.color_imgs_dir):
        for each_sub_dir in sub_dir:
            out_root_dir = root.replace(args.color_imgs_dir, args.gray_imgs_dir)
            out_dir = os.path.join(out_root_dir, each_sub_dir)
            os.makedirs(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--color_imgs_dir',
        type=str,
        default=color_imgs_dir)
    parser.add_argument(
        '--gray_imgs_dir',
        type=str,
        default=gray_imgs_dir)
    args_ = parser.parse_args()
    if os.path.exists(args_.gray_imgs_dir):
        shutil.rmtree(args_.gray_imgs_dir)
    os.makedirs(args_.gray_imgs_dir)
    main(args_)
