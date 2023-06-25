#!/bin/env python
import argparse
import os
import gdown

PARSER = argparse.ArgumentParser(description="galactic_rings")
PARSER.add_argument('--data_dir',
                    type=str,
                    default='data/images_train',
                    help="""Directory where to download the dataset""")

def main():
    FLAGS = PARSER.parse_args()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filename = 'rings_nonrings.zip'

    file_id='1GH0qk0n_6gG7sCqpcIFKdVMMilIlZj64'
    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, os.path.join(FLAGS.data_dir, filename), quiet=False)

    print("Finished downloading files for galaxy detection into {}".format(FLAGS.data_dir))


if __name__ == '__main__':
    main()
