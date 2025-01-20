# pyranges0
import os
import shutil
import zipfile
from pathlib import Path

import emoji
import gdown
import pyranges as pr0
import pyranges1 as pr1
from tqdm import tqdm

from logger import logger


def df2pr0(df):
    return pr0.PyRanges(
        chromosomes=df.contig,
        starts=df.pos_start,
        ends=df.pos_end,
    )


### pyranges1
def df2pr1(df):
    return pr1.PyRanges(
        {
            "Chromosome": df.contig,
            "Start": df.pos_start,
            "End": df.pos_end,
        }
    )


def prepare_datatests(datasets, BECH_DATA_ROOT):
    global dataset, file
    Path.mkdir(Path(BECH_DATA_ROOT), parents=True, exist_ok=True)
    for d in tqdm(datasets, desc="Downloading benchmark datasets"):
        url = d["url"]
        dataset = d["name"]
        file = f"{dataset}.zip"
        extract_dir = f"{BECH_DATA_ROOT}/"
        if os.path.exists(f"{extract_dir}/{dataset}"):
            logger.info(
                emoji.emojize(
                    f"Dataset {file.split(".")[0]} already exists :file_folder: To re-download, remove the directory."
                )
            )
            continue
        gdown.download(url, f"{BECH_DATA_ROOT}/{file}", quiet=True, verify=True)
        logger.info(emoji.emojize(f"Downloaded {file} :check_box_with_check: "))
        ds_zip_file = f"{BECH_DATA_ROOT}/{file}"
        Path.mkdir(Path(extract_dir), parents=True, exist_ok=True)
        shutil.rmtree(f"{extract_dir}/*", ignore_errors=True)
        with zipfile.ZipFile(ds_zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(emoji.emojize(f"Extracted {file} :open_file_folder: "))
        os.remove(ds_zip_file)
    logger.info(emoji.emojize("Downloaded all benchmark datasets :check_mark_button:"))
