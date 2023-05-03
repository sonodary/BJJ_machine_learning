# Script/Package to load batches
# PACKAGES
import gdown
import logging
from zipfile import ZipFile
import os
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import concurrent.futures
import cv2

# GLOBAL VARIABLES
batchIds = {
    1: "1p-7tN3D3g80GQoIR8tYIgzQUMTqNWwGZ",
    2: "1BDwQ8VGLXKCv7GWSKGlCtfFs-kCokwGW",
    3: "1EAQiIiOsk1ZILaXN-cmbZ6xw1GsHdOtj",
    4: "1upmzQCmro-wRADEXpUKTk1kwudgkJqJi",
    5: "1U4KKHJV3QqjG1Z8ZdKAnRD98m2rQRWBC",
    6: "1w0ig4Z1SaQVi8RIZvYXTr7zsZwyVPJbz",
    7: "1ublw71KXeRLMSuxNFxnmVA4z36v-PXN6",
    8: "1ft0HPxsuIdRIxDqUY3akKHJABYnLpvP8",
    9: "1qi-D3CkHcIXVIFslmlz6lVa2cQsMjFQA",
    10: "1rbeMzL7AS0sCf8reVVyTPF6G_aPrMdFS",
    11: "1-wA2Lw89Uii70ckW8tj10heDSaKnuQYS",
    12: "1vRNwzXx-ZvMKHQXYMGLt4zkbDmt85Jm4",
}

IMG_HEIGHT = 64
IMG_WIDTH = 64


# FUNCTIONS
def getBatch(num: int, path: str, unzip: bool = False) -> None:
    """Fetches the batch number `num` from the google drive folder with URL `url`.
        Writes the file to `path`.

    Args:
        num (int): number of desired batch
        url (str): url that points to google drive folder
        path (str): path to output file for the desired batch. Should be in format: './batch.zip'
    """
    if ".zip" not in path:
        logging.error('Specified batch output path does not end with ".zip"')
        return

    # Setting up URL for request
    batchId = batchIds[num]
    url = f"https://drive.google.com/file/d/{batchId}/view?usp=sharing"

    # Requesting file from Google Drive
    logging.info(f"Downloading batch {num}")
    gdown.download(url, path, quiet=False, fuzzy=True)

    # Unzip
    if unzip:
        unzipBatch(path)


def getBatches(batchDict: dict, unzip: bool = False) -> None:
    """Gets multiple batches from drive. Takes a dict in the form {batchNum: 'output/path'}

    Args:
        batchDict (dict): dict of batch numbers and output paths
    """
    for item in batchDict.items():
        getBatch(item[0], item[1], unzip)


def unzipBatch(batchPath: str, outputPath: str = "") -> None:
    """Unzips batch at `batchPath` to folder at `outputPath` if given.

    Args:
        batchPath (str): path to batch
        outputPath (str, optional): Optional argument if you want to specify output file path. Defaults to "".
    """
    # If no output path is given, then unzip to folder with same file name in the same directory.
    if outputPath == "":
        outputPath = batchPath.replace(".zip", "")

    with ZipFile(batchPath, "r") as f:
        # If output folder doesn't exist, create it
        if not os.path.exists(outputPath):
            logging.info(
                f"Output Folder not found! Creating output folder at {outputPath}."
            )
            os.makedirs(outputPath)

        # Unzip to output folder
        logging.info(f"Extracting {batchPath} to {outputPath}")
        f.extractall(outputPath)


def importImages(filePath: str) -> tuple:
    """Takes a path to a file of images and returns a numpy array of image tensors and a list of their ids.
        Takes at most a couple of minutes. Imports images in parallel because it was taking too long with a single thread.

    Args:
        filePath (str): filepath to folder with images. Must end with "/"

    Returns:
        tuple(list, np.array): tuple containing a list of ids and an np.array with their respective image tensors.
    """
    # Use Pathlib to avoid string concatenation
    path = Path(filePath)

    # Get list of image file names
    imgNames = [f.name for f in path.glob("*.jpg")]

    # Pre-allocate output arrays
    images = np.empty((len(imgNames), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    imgIds = np.empty(len(imgNames), dtype=np.int64)

    # Define a function to load a single image
    def load_single_image(idx, name):
        # Finding image id
        imgIds[idx] = int(name.replace(".jpg", ""))

        # Opening image with OpenCV
        imgPath = str(path / name)
        rawImage = cv2.imread(imgPath, cv2.IMREAD_COLOR)

        # Resizing to desired height and width with INTER_AREA method
        resizedImage = cv2.resize(
            rawImage, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
        )

        # Saving to output array
        images[idx] = resizedImage

    # Use a thread pool to parallelize image loading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the image loading tasks to the pool
        futures = [
            executor.submit(load_single_image, idx, name)
            for idx, name in enumerate(imgNames)
        ]
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    return (np.array(imgIds), images)


def loadImageBatch(num: int) -> tuple:
    """Downloads and converts an image batch to two np.arrays one of the ids and one of the image tensors.
        USE THIS FUNCTION FOR MOST THINGS.

    Args:
        num (int): number of batch

    Returns:
        tuple: tuple of np.arrays. First one is ids, second is image tensors
    """
    path = f"./batch{num}.zip"
    # We download and unzip batch from drive
    getBatch(num, path, unzip=True)

    # We import images into np.arrays
    newPath = path.replace(".zip", f"/part_{num}")
    ids, images = importImages(newPath)

    return (ids, images)


def loadImageBatches(nums: list) -> tuple:
    """Downloads and converts multiple image batches to two np.arrays one of the ids and one of the image tensors.
        Not completely sure if it works, but it should.

    Args:
        nums (int): number of batch

    Returns:
        tuple: tuple of np.arrays. First one is ids, second is image tensors
    """
    idArrays, imageArrays = [], []
    for num in nums:
        ids, images = loadImageBatch(num)

        # Adding to arrays
        idArrays.append(ids)
        imageArrays.append(images)

    totalIds = np.concatenate(idArrays) if len(idArrays) > 1 else idArrays[0]
    totalImages = (
        np.concatenate(imageArrays) if len(imageArrays) > 1 else imageArrays[0]
    )

    return (totalIds, totalImages)


if __name__ == "__main__":
    ids, images = loadImageBatches([1])
