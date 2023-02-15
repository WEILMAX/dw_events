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
    location: str = 'scbalm'
    
    def __post_init__(self):
        self.api:API =  API(api_root=self.root, username=self.user, password=self.password)
    
    def get_metrics(self) -> list[str]:
        """_summary_

        Returns:
            _type_: _description_
        """
        self.metrics = self.api.metrics(locations=[self.location])
        return self.metrics

    def select_metrics(
        self,
        selection_crietria:list[str]
        ) -> list[str]:
        """_summary_

        Args:
            selection_crietria (list[str]): _description_

        Returns:
            _type_: _description_
        """
        self.get_metrics()
        metrics = self.metrics
        selected_metrics = metrics['metric']
        for selection_crietrium in selection_crietria:
            selected_metrics = [m for m in selected_metrics if selection_crietrium in m]
        return selected_metrics

    def get_fbg_metrics(
        self,
        selection_criteria = ['FBG']
        ) -> list[str]:
        fbg_metrics = \
            self.select_metrics(
                selection_crietria=selection_criteria
            )
        return fbg_metrics

    