"""
A class that holds the parameters and associated functions."""
import logging

log = logging.getLogger(__name__)


class Parameters:
    """
    Container for fit-configuration parameters.

    This class wraps the ``parameters`` section of a freqfit YAML configuration and
    provides helpers to extract the subset of parameters used by a set of
    :class:`~freqfit.dataset.Dataset` objects.

    Parameters
    ----------
    parameters
        Mapping from parameter name to a parameter specification dictionary.

    Attributes
    ----------
    parameters
        The raw parameter configuration.
    """
    def __init__(
        self,
        parameters: dict,
    ) -> None:
        """
        Takes parameters dict from config and holds it as an object.
        """

        self.parameters = parameters

        # get the parameters of interest
        self.poi = []
        for parname, par in self.parameters.items():
            if "poi" in par.keys():
                self.poi.append(parname)

                msg = f"added '{parname}' as parameter of interest"
                logging.info(msg)

    def __call__(
        self,
        par: str,
    ) -> dict:
        """
        Return the configuration dictionary for a given parameter name.

        Parameters
        ----------
        par
            Parameter name.

        Returns
        -------
        dict
            Parameter configuration.
        """
        return self.parameters[par]

    def get_parameters(
        self,
        datasets: dict["Dataset"],  # noqa: F821
        nodata: bool = False,
    ) -> dict:
        """
        Takes dict of Dataset and returns all parameters used in them as a dict.

        Parameters
        ----------
        datasets 
            dict of Dataset
        nodata : bool
            If `True`, returns parameters of passed datasets that have no data (are empty)
        """

        allpars = set()
        parswdata = set()
        for ds in datasets.values():
            allpars.update(ds._parlist)

            if ds.data.size > 0:
                parswdata.update(ds._parlist)

        if nodata:
            parsnodata = allpars.difference(parswdata)
            return {p: self.parameters[p] for p in list(parsnodata)}

        return {p: self.parameters[p] for p in list(allpars)}

    def get_fitparameters(
        self,
        datasets: dict["Dataset"],  # noqa: F821
        nodata: bool = False,
    ) -> dict:
        """
        Takes dict of Dataset and returns all fit parameters used in them as a dict.

        Parameters
        ----------
        datasets 
            dict of Dataset
        nodata : bool
            If `True`, returns parameters of passed datasets that have no data (are empty)
        """

        allpars = set()
        parswdata = set()
        for ds in datasets.values():
            allpars.update(list(ds.fitparameters.keys()))

            if ds.data.size > 0:
                parswdata.update(list(ds.fitparameters.keys()))

        if nodata:
            parsnodata = allpars.difference(parswdata)
            return {p: self.parameters[p] for p in list(parsnodata)}

        return {p: self.parameters[p] for p in list(allpars)}
