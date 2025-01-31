import argparse
import sys
import os
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

import imagehash
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS=None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def calculate_signature(image_file: str, hash_size: int) -> np.ndarray:
    """ 
    Calculate the dhash signature of a given file
    
    Args:
        image_file: the image (path as string) to calculate the signature for
        hash_size: hash size to use, signatures will be of length hash_size^2
    
    Returns:
        Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
    """
    pil_image = Image.open(image_file).convert("L").resize(
                        (hash_size+1, hash_size),
                        Image.ANTIALIAS)
    dhash = imagehash.dhash(pil_image, hash_size)
    signature = dhash.hash.flatten()
    pil_image.close()
    return signature

        
def find_near_duplicates(input_dir: str, threshold: float, hash_size: int, bands: int) -> List[Tuple[str, str, float]]:
    """
    Find near-duplicate images
    
    Args:
        input_dir: Directory with images to check
        threshold: Images with a similarity ratio >= threshold will be considered near-duplicates
        hash_size: Hash size to use, signatures will be of length hash_size^2
        bands: The number of bands to use in the locality sensitve hashing process
        
    Returns:
        A list of near-duplicates found. Near duplicates are encoded as a triple: (filename_A, filename_B, similarity)
    """
    input_dir = input_dir.replace('\"', "")
    rows: int = int(hash_size**2/bands)
    signatures = dict()
    hash_buckets_list: List[Dict[str, List[str]]] = [dict() for _ in range(bands)]
    
    # Build a list of candidate files in given input_dir
    file_list = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]

    # Iterate through all files in input directory
    for fh in file_list:
        try:
            signature = calculate_signature(fh, hash_size)
        except IOError:
            # Not a PIL image, skip this file
            continue

        # Keep track of each image's signature
        signatures[fh] = np.packbits(signature)
        
        # Locality Sensitive Hashing
        for i in range(bands):
            signature_band = signature[i*rows:(i+1)*rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in hash_buckets_list[i]:
                hash_buckets_list[i][signature_band_bytes] = list()
            hash_buckets_list[i][signature_band_bytes].append(fh)

    # Build candidate pairs based on bucket membership
    candidate_pairs = set()
    for hash_buckets in hash_buckets_list:
        for hash_bucket in hash_buckets.values():
            if len(hash_bucket) > 1:
                hash_bucket = sorted(hash_bucket)
                for i in range(len(hash_bucket)):
                    for j in range(i+1, len(hash_bucket)):
                        candidate_pairs.add(
                            tuple([hash_bucket[i],hash_bucket[j]])
                        )

    # Check candidate pairs for similarity
    near_duplicates = list()
    for cpa, cpb in candidate_pairs:
        hd = sum(np.bitwise_xor(
                np.unpackbits(signatures[cpa]), 
                np.unpackbits(signatures[cpb])
        ))
        similarity = (hash_size**2 - hd) / hash_size**2
        if similarity > threshold:
            near_duplicates.append((cpa, cpb, similarity))
            
    # Sort near-duplicates by descending similarity and return
    near_duplicates.sort(key=lambda x:x[2], reverse=True)
    return near_duplicates


def main():
    # Argument parser
    #parser = argparse.ArgumentParser(description="Efficient detection of near-duplicate images using locality sensitive hashing")
    #parser.add_argument("-i", "--inputdir", type=str, default="", help="directory containing images to check")
    #parser.add_argument("-t", "--threshold", type=float, default=0.9, help="similarity threshold")
    #parser.add_argument("-s", "--hash-size", type=int, default=16, help="hash size to use, signature length = hash_size^2", dest="hash_size")
    #parser.add_argument("-b", "--bands", type=int, default=16, help="number of bands")
    #parser.add_argument("-o", "--outdir", type=str, default="", help="Enter output csv dictory ex.원하는 경로/csv파일명.csv")

    #args = parser.parse_args()
    input_dir = input('input path')
    threshold = float(input('threshold number'))
    hash_size = 16
    bands = 16
    save = input('save path')
    save = save.replace('\"', "")
                 
    key_list = []
    value_list = []

    try:
        #print('this way')
        #print('dir', input_dir)
        near_duplicates = find_near_duplicates(input_dir, threshold, hash_size, bands)
        if near_duplicates:
            print(f"Found {len(near_duplicates)} near-duplicate images in {input_dir} (threshold {threshold:.2%})")
            for a,b,s in near_duplicates:
                # a -> 파일 1 경로 / b -> 파일 2 경로 / s -> 유사도 | s를 key로 a,b를 value로
                print(f"{s:.2%} similarity: file 1: {a} - file 2: {b}")
                key_list.append(s)
                value = [a, b]
                value_list.append(value)
        else:
            print(f"No near-duplicates found in {input_dir} (threshold {threshold:.2%})")
    except OSError:
        print(f"Couldn't open input directory {input_dir}")    #여기로 빠지는데 이유가 뭐지?
    
    pair = defaultdict(list)
    for i in range(len(key_list)):
        pair[key_list[i]].append(value_list[i])
    
    #print('??', pair)
    
    value_len = []
    for k,v in pair.items():
        len_v = len(v)
        value_len.append(len_v)
    
    #print('tt', value_len)
    
    max_len = max(value_len)
    
    for k,v in pair.items():
        if len(v) < max_len:
            pad = max_len - len(v)
            for i in range(pad):
                v.append('Nan')
             
    df = pd.DataFrame.from_dict(pair)
    df.to_csv(save, index = True, header =True)
    

if __name__ == "__main__":
    main()
