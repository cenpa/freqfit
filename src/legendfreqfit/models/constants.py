import numpy as np

QBB = 2039.0612  # 2039.0612 +- 0.0075 keV from AME2020
NA = 6.0221408e23  # Avogadro's number
M76 = 0.0759214027  # kilograms per mole, molar mass of 76Ge
MDET = 0.075681  # kg/mol, average molar mass for detectors from MJD Unidoc # M-TECHDOCUNIDOC-2022-068
G_01 = 0.23e-14  # in yr^-1, phase space factor for 76Ge from Phys. Rev. C 98, 035502 by Horoi et. al
g_A = 1.27  # noqa: N816 # axial coupling constant from Phys. Rev. Lett. 120, 202002 by Czarnecki et. al
me = 0.5109989500e6  # mass of the electron in eV/c^2 from PDG
NME_central = 2.60  # these values come from Phys. Rev. Lett. 132, 182502
NME_unc_hi = 1.28  # symmetrized error from Phys. Rev. Lett. 132, 182502
NME_unc_lo = 1.36  # symmetrized error from Phys. Rev. Lett. 132, 182502
# NME_PHENO_LOW = 2.12  # lower phenomenological  NME published after the US nuclear physics long range plan, Jokiniemi et al. https://doi.org/10.1103/PhysRevC.107.044305
NME_PHENO_LOW = 2.35  # lower phenomenological NME from the US nuclear physics long range plan arXiv:2212.11099, Jiao et al. doi.org/10.1103/PhysRevC.96.054310
NME_PHENO_HIGH = 6.34  # lower phenomenological NME from the US nuclear physics long range plan arXiv:2212.11099, Deppisch et al. doir.org/10.1103/PhysRevD.102.095016
# NME_PHENO_HIGH = 6.79  # upper phenomenological  NME published after the US nuclear physics long range plan, Jokiniemi et al. https://doi.org/10.1103/PhysRevC.107.044305

# lines to exclude are
# 2614.511(10) - 511 = 2103.511 keV SEP from 208Tl
# 2118.513(25) keV from 214Bi
# 2204.10(4) keV from 214Bi
# then round to nearest 0.1 keV

# default analysis window (in keV)
WINDOW = [[1930.0, 2098.5], [2108.5, 2113.5], [2123.5, 2190.0]]

# MJD analysis window (in keV) is slightly larger than GERDA/LEGEND and excludes an additional line
MJD_WINDOW = [
    [1950.0, 2098.5],
    [2108.5, 2113.5],
    [2123.5, 2199.1],
    [2209.1, 2350.0],
]

# could use these to go a little faster?
LOG2 = 0.69314718055994528622676398299518041312694549560546875
SQRT2PI = 2.506628274631000241612355239340104162693023681640625


# conversion function
def s_prime_to_s(s_prime):
    # Given s_prime in decays/(kg*yr), find s in decays/yr
    s = s_prime * (MDET / (LOG2 * NA))
    return s


# conversion function
def s_to_s_prime(s):
    # Given s in decays/yr, find s_prime in decays/(kg*yr)
    s_prime = s / (MDET / (LOG2 * NA))
    return s_prime


def halflife_to_s_prime(halflife):
    return s_to_s_prime(1.0 / halflife)


# another conversion function
def m_prime_to_m(m_prime):
    # Given m_prime in ev/(kg*yr), find m in ev
    return m_prime * np.sqrt(MDET * me**2 / (LOG2 * NA * G_01 * g_A**4))


def s_prime_to_halflife(s_prime):
    # Given s_prime in decays/(kg*yr), find t_half in yrs
    t_half = 1 / (s_prime_to_s(s_prime))
    return t_half
