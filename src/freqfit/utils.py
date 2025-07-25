import inspect
import logging

import numpy as np
import yaml

log = logging.getLogger(__name__)

# negative of the exponent of scientific notation of a number
def negexpscinot(number):
    base10 = np.log10(abs(number))
    return int(-1 * np.floor(base10))
