# -*- coding: utf-8 -*-
import datetime
import os
from typing import Union
from azure.storage.blob import BlobServiceClient
from dotenv import find_dotenv, load_dotenv
from utils import make_dt_list
from dataclasses import dataclass


@dataclass
class DataGetter:
    """
    A class for getting data from a database.
    """
    start: datetime.datetime
    stop: datetime.datetime
    container_name: str = 'data-primary-smartbridge'
    connection_string_name: str = 'SCB_CONNECTION_STR'


    def __post_init__(self):
        load_dotenv(find_dotenv())
        self.connect_str: Union[str, None] = \
            os.getenv(self.connection_string_name)
        if self.connect_str is None:
            raise \
                ValueError(
                    f'{self.connection_string_name}\
                        not found as environmental variable.'
                    )
        else:
            self.blob_service_client: BlobServiceClient = \
                BlobServiceClient.from_connection_string(
                    self.connect_str)
            self.container_client = \
                self.blob_service_client.get_container_client(
                    container=self.container_name)
    

    def get_strain_data(self, interval:int = 600):
        """
        Get strain data from the database.
        """
        pass
