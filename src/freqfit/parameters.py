"""
A class that holds the parameters and associated functions.
"""
import logging

SEED = 42

log = logging.getLogger(__name__)

class Parameters:

    def __init__(
        self,
        parameters: dict,
    ) -> None:
        """
        Takes parameters dict from config and holds it as an object.
        """

        self.parameters = parameters

        return None

    def __call__(
        self,
        par: str,
    ) -> dict:

        return self.parameters[par]
    
    def get_parameters(
        self,
        datasets: dict,
        nodata: bool = False,
    ) -> dict:
        """
        Takes dict of Dataset and returns all parameters used in them as a dict.
        """

        allpars = set()
        parswdata = set()
        for ds in datasets.values():
            allpars.update(ds._parlist)

            if (ds.data.size > 0):
                parswdata.update(ds._parlist)
        
        if nodata:
            parsnodata = allpars.difference(parswdata)
            return {p:self.parameters[p] for p in list(parsnodata)}

        return {p:self.parameters[p] for p in list(allpars)}

    def get_fitparameters(
        self,
        datasets: dict,
        nodata: bool = False,
    ) -> dict:
        """
        Takes dict of Dataset and returns all fit parameters used in them as a dict.
        """

        allpars = set()
        parswdata = set()
        for ds in datasets.values():
            allpars.update(list(ds.fitparameters.keys()))

            if (ds.data.size > 0):
                parswdata.update(list(ds.fitparameters.keys()))
        
        if nodata:
            parsnodata = allpars.difference(parswdata)
            return {p:self.parameters[p] for p in list(parsnodata)}

        return {p:self.parameters[p] for p in list(allpars)}

