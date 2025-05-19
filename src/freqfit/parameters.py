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

        if type(par) is str:
            return self.parameters[par]
    
    def get_parameters(
        self,
        datasets: dict,
    ) -> dict:
        """
        Takes dict of Dataset and returns all parameters used in them as a dict.
        """

        allpars = set()
        for ds in datasets.values():
            allpars.update(ds._parlist)

        return {p:self.parameters[p] for p in list(allpars)}


