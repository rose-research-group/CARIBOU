from abc import ABC, abstractmethod
import json
import numba
numba.config.CACHE = False
import os
os.environ['NUMBA_CACHE_OVERRIDE'] = '0'

class AutoMetric(ABC):
    """
    Abstract base class for a metric to be applied to an AnnData object.
    """ 
    @abstractmethod
    def metric(self, adata) -> dict:
        """
        Run the metric and return a dictionary of results.
        """
        pass

    @abstractmethod
    def requirements(self) -> str:
        """
        Return a description of the metric.
        """
        pass
    
    def run(self, adata):
        """
        Handles execution + JSON serialization.
        """
        try:
            result = self.metric(adata)
            print(json.dumps(result))  # Always print result at the end
        except:
            result = {"Not ready yet": self.requirements()}
            print(json.dumps(result))