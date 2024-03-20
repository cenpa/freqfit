"""
Models for unbinned frequentist fitting. These models are classes and come with methods to compute the pdf and logpdf.
"""

from legendfreqfit.models.gaussian_on_uniform import gaussian_on_uniform

__all__ = ["gaussian_on_uniform"]
