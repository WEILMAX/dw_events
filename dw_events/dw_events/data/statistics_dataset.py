"""
statistics_dataset.py is a module for gettign the 1 or 10min statistics
through the owi-lab api.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
# -*- coding: utf-8 -*-
import datetime
import os
from dataclasses import dataclass
from owi_data_2_pandas.io import API
from typing import cast


@dataclass
class StatisticsGetterSCB:
    start: datetime.datetime
    stop: datetime.datetime
    user: str = 'maximillian.weil@vub.be'
    password: str = cast(str, os.getenv('SCB_API_PASSWORD'))
    root: str = r"https://api.smartcircularbridge.eu/api/v1/"
    

    def __post_init__(self):
        self.api:API =  API(api_root=self.root, username=self.user, password=self.password)

    def get_statistics(self):
        new_data = self.api.query(
                        period,
                        location=self.location,
                        metrics=metrics
                        ).resample(
                            sample_freq,
                            offset='5T'
                            ).last().resample(
                                sample_freq,
                                closed='left',
                                label='right'
                                ).last()#.resample(sample_freq, closed='right').last()