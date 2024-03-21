from abc import ABC, abstractmethod

import geopandas as gpd


class PatchStatsAdder(ABC):
    @abstractmethod
    def process(self, geo_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        pass
