# crisp packages
# TODO: convert to package

# misc packages
import os
import random
import requests
import datetime
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing


## ---------- Helper fns ---------- ##


# https://stackoverflow.com/questions/2659900/slicing-a-list-into-n-nearly-equal-length-partitions
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

def check_image_valid(savepath):
    # check and see if it actually saved properly w/ pil verify
    try:
        i = Image.open(savepath).convert('RGB')
        return 'yes'
    except:
        return 'no'
    

def download_image(savepath, image):
    download_success = 'yes'
    # open file
    with open(savepath, 'wb') as f:
        # try downloading image
        try:
            r = requests.get(image)
            if r.status_code == requests.codes.ok:
                f.write(r.content)
            else:
                download_success = 'no'
        except:
            download_success = 'no'

    # check and see if it actually saved properly w/ pil verify
    if download_success != 'no':
        download_success = check_image_valid(savepath)
    return download_success
    

## ---------- Scraper code ---------- ##
    
def download_images(obs, split, data_path):

    # use good ol requests to downlod the images one-by-one
    # https://inaturalist-open-data.s3.amazonaws.com/photos/226405959/small.jpeg
    obs['download_success'] = 'yes'
    for i, row in tqdm(obs.iterrows(), total=len(obs)):
        image = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{row.photo_id}/small.{row.extension}"
        savedir = f"{data_path}{split}/{str(row.photo_id)[:3]}/"
        savepath = f"{savedir}{row.photo_id}.png"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # first, check if image exists and is uncorrupted
        download_success = check_image_valid(savepath)
        # if not already downloaded, grab the image
        if download_success != 'yes':
            download_success = download_image(savepath, image)
        # save if image is good or not
        obs.at[i, 'download_success'] = download_success
    
    obs = obs[obs.download_success == 'yes']
    return obs


def download_images_parallel(obs, procid, lock, split, data_path):
    obs['download_success'] = 'yes'
    with lock:
        prog = tqdm(total=len(obs), desc=f"downloading images for proc {procid}", unit=' observations', position=procid)
    for i, row in obs.iterrows():
        image = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{row.photo_id}/small.{row.extension}"
        savedir = f"{data_path}{split}/{str(row.photo_id)[:3]}/"
        savepath = f"{savedir}{row.photo_id}.png"
        if not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)
            
        # first, check if image exists and is uncorrupted
        download_success = check_image_valid(savepath)
        # if not already downloaded, grab the image
        if download_success != 'yes':
            download_success = download_image(savepath, image)
        # save if image is good or not
        obs.at[i, 'download_success'] = download_success
        with lock:
            prog.update(1)
    return obs

## ---------- main loop ---------- ##

def execute_download(args):
    
    
    # data_path = f"{args.data_dir}{args.save_dir}/"
    obs = pd.read_csv(f"{args.data_dir}{args.split}/observations_cleaned_preimage.csv") 
    
    print(f"saving inat photos to {args.data_dir}{args.split}/ directory")
    # serial download
    if args.parallel < 2:
        obs = download_images(obs, args.split, args.data_dir)
    else:
        # parallel download
        # now, chunk up the dataset into K sections
        if args.shuffle:
            idxs = list(range(len(obs)))
            random.shuffle(idxs)
        else:
            idxs = list(range(len(obs)))
        idx_pars = partition(idxs, args.parallel)
        procs = []
        # TQDM for parallel processes: https://stackoverflow.com/questions/66208601/tqdm-and-multiprocessing-python
        lock = multiprocessing.Manager().Lock()
        pool =  multiprocessing.Pool(args.parallel)

        res_async = [pool.apply_async(download_images_parallel, args=(obs.iloc[idxs], i, lock, args.split, args.data_dir)) for i, idxs in enumerate(idx_pars)]
        res_dfs = [r.get() for r in res_async]
        pool.close()
        pool.join()
        obs = pd.concat(res_dfs)
        obs = obs[obs.download_success == 'yes']
    print(f"saving to {args.data_dir}{args.split}/observations_postGL.csv")
    obs.to_csv(f"{args.data_dir}{args.split}/observations_postGL.csv", index=False)

## ---------- args setup ---------- ##

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--split', type=str, required=True, help='Which dataset to download', choices=['alaska', 'british_columbia', 'washington', 'oregon', 'california', 'baja_california', 'baja_california_sur', 'yukon', 'nevada', 'arizona', 'sonora']) 
    args.add_argument('--parallel', type=int, required=True, help='num. cores to use')
    # TODO: convert to paths.DATA?s
    args.add_argument('--data_dir', type=str, help='what directory to look and save to', required=True) 
    args.add_argument('--shuffle', action='store_true', help='whether to shuffle the images across parallel processes (helpful if a few threads crashed)')

    args, _ = args.parse_known_args()
    execute_download(args)