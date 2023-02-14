# -*- coding: utf-8 -*-
import datetime
import os
import pandas as pd
from dataclasses import dataclass
from typing import Union
from azure.storage.blob import BlobServiceClient
from dotenv import find_dotenv, load_dotenv
from .utils import make_dt_list, copy_gzip_to_local


@dataclass
class DataGetter:
    """
    A class for getting data from a blob storage.
    """

    start: datetime.datetime
    stop: datetime.datetime
    container_name: str = "data-primary-smartbridge"
    connection_string_name: str = "SCB_CONNECTION_STR"
    merged_signals: pd.DataFrame = pd.DataFrame()

    def __post_init__(self):
        """Initializes the DataGetter class after initialization.
        Loads the connection string from the environment variable
        and creates a BlobServiceClient object.
        """
        load_dotenv(find_dotenv())
        self.connect_str: Union[str, None] = os.getenv(self.connection_string_name)
        if self.connect_str is None:
            raise ValueError(
                f"{self.connection_string_name}\
                        not found as environmental variable."
            )
        else:
            self.blob_service_client: BlobServiceClient = (
                BlobServiceClient.from_connection_string(self.connect_str)
            )
            self.container_client = self.blob_service_client.get_container_client(
                container=self.container_name
            )

    def get_strain_data(
        self,
        interval: int = 600,
        filetype: str = "fiber",
        location: str = "SCBALM",
        destination: str = "../data/raw/SCBALM",
    ) -> pd.DataFrame:
        """Retrieves strain data from the blob storage and returns it as a DataFrame.
        The function retrieves data from the blob storage for the given time interval, location and filetype.
        It then converts the data into a list of signals, merges the signals and returns it as a DataFrame.

        Args:
            interval (int, optional): The interval in seconds between two consecutive data points.
                Defaults to 600.
            filetype (str, optional): The type of file to be retrieved.
                Defaults to 'fiber'.
            location (str, optional): The location of the data.
                Defaults to 'SCBALM'.
            destination (str, optional): The local destination path for the data.
                Defaults to '../data/raw/SCBALM'.

        Returns:
            pd.DataFrame: Dataframe containing the strain signals with columns:
        """
        dt_list = make_dt_list(self.start, self.stop, interval)
        merged_signals = copy_gzip_to_local(
            dt_list[0],
            self.blob_service_client,
            self.container_name,
            filetype,
            location,
            destination,
        )
        for date_time in dt_list:
            try:
                new_signals = copy_gzip_to_local(
                    date_time,
                    self.blob_service_client,
                    self.container_name,
                    filetype,
                    location,
                    destination,
                )
                merged_signals = pd.concat([merged_signals, new_signals], axis=0)
            except FileNotFoundError as exc:
                print(exc, ". Failed to import and merge strain signal at ", datetime)
        self.merged_signals = merged_signals
        return merged_signals

    def get_dataframe_str_subset(self, strain_line: str) -> pd.DataFrame:
        """get a subset of a pandas dataframe with a string in column name

        Args:
            strain_line (str): Name of the strain line to select
        """
        if len(self.merged_signals) == 0:
            self.get_strain_data()
        return self.merged_signals.filter(regex=f"{strain_line}")


