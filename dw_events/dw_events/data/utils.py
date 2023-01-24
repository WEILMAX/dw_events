"""
utils.py is a module for helper functions in the dw_events.data subpackage.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
# Global import
import datetime
import os
from typing import List
from azure.storage.blob import BlobServiceClient
import pytz
import pandas as pd

# Imports from internal packages
from dw_signal import fbgs, signal


def make_dt_list(
    start_dt: datetime.datetime, stop_dt: datetime.datetime, interval=60
) -> List[datetime.datetime]:
    """Generates a list of datetime timestamps
    between a start_dt and a stop_dt

    Args:
        start_dt (datetime.datetime): Starting timestamp of the list
        stop_dt (datetime.datetime): Final timestamp of the list
        interval (int, optional): Interval in seconds between each step.
            Defaults to 60.

    Returns:
        _type_: _description_
    """
    nr_steps = int((stop_dt - start_dt).total_seconds() / interval)
    dt_list = [start_dt + datetime.timedelta(0, interval * x) for x in range(nr_steps)]
    return dt_list


def copy_gzip_to_local(
    date_time: datetime.datetime,
    blob_service_client: BlobServiceClient,
    container_name: str,
    filetype="fiber",
    location="SCBALM",
    destination="../data/raw/SCBALM",
) -> pd.DataFrame:

    """Copies a gzip file from a blob storage and outputs it as a SignalList.
    The function takes a date_time and converts it to the local timezone,
    then uses it to construct the path of the file in the blob storage.
    The function then creates a blob client, and downloads the file from the blob storage
    to the local destination. If the file already exists in the local destination,
    it will not download it again.
    Finally, the function reads the file, and outputs it as a SignalList.

    Args:
        date_time (datetime.datetime): A datetime object,
            representing the date and time of the file to be downloaded.
        blob_service_client (BlobServiceClient): A blob service client object,
            to interact with the blob storage.
        container_name (str): The name of the container in the blob storage.
        filetype (str, optional): Type of file to look for (fiber, acc or acc_env).
            Defaults to 'fiber'.
        location (str, optional): Location of the structure.
            Defaults to 'SCBALM'.
        destination (str, optional): Local destination path for the file.
            Defaults to '../data/raw/SCBALM'.

    Returns:
        pd.DataFrame: A timeseries object, containing the data from the file.
    """
    dt_local = date_time.astimezone(pytz.timezone("Europe/Amsterdam"))
    utc_offset_hours = int(
        date_time.astimezone(pytz.timezone("Europe/Amsterdam")).utcoffset().seconds
        / 3600
    )
    azure_path = r"/".join(
        [
            str(location),
            "TDD",
            "TDD_" + filetype,
            dt_local.strftime("%Y/%m/%d/%Y%m%d-%H%M")
            + " 0"
            + str(utc_offset_hours)
            + "-SCB.txt.gz",
        ]
    )
    local_path = os.path.join(
        destination,
        str(location),
        "TDD",
        "TDD_" + filetype,
        dt_local.strftime("%Y/%m/%d/%Y%m%d-%H%M") + " 02-SCB.txt.gz",
    )
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=azure_path
    )
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())  # type: ignore
        except FileNotFoundError as exc:
            print("File not found:", exc, f". Failed to download {azure_path}")
            os.remove(local_path)
    strain_signal = fbgs.read_fbgs(local_path, location=location)
    return strain_signal.as_df(style="timeseries")  # type: ignore


def get_dataframe_str_subset(
    data: pd.DataFrame,
    strain_line: str
    ) -> pd.DataFrame:
    """get a subset of a pandas dataframe with a string in column name

    Args:
        data (pd.DataFrame): dataframe with strain data
        strain_line (str): Name of the strain line to select
    """
    return data.filter(regex=f'{strain_line}')