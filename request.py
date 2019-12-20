import requests
import pandas as pd
from multiprocessing.pool import ThreadPool
from itertools import product
from functools import partial
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--doc", required=True,
                help="path to csv")
ap.add_argument("-i", "--imagePath", required=True, help="path to csv")
ap.add_argument("-if", "--imageFetchUrl", required=True, help="path to csv")
args = vars(ap.parse_args())


def fetch_image(item_image_path):
    url = args["imageFetchUrl"] + item_image_path
    r = requests.get(url, allow_redirects=True)
    return r


def fetch_image_parallel(df, threads):
    pool = ThreadPool(threads)
    item_image_path = df['item_image']
    results = pool.map(fetch_image, item_image_path)
    pool.close()
    pool.join()
    return results


if __name__ == "__main__":
    df = pd.read_csv(args['doc'])
    imageResults = fetch_image_parallel(df, 10)
    for i in range(len(imageResults)):
        item_pk = df['item_pk']
        imgPath = args['imagePath'] + item_pk[i] + '.jpg'
        open(imgPath, 'wb').write(imageResults[i].content)
