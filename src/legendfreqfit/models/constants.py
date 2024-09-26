QBB = 2039.0612  # 2039.0612 +- 0.0075 keV from AME2020
NA = 6.0221408e23  # Avogadro's number
M76 = 0.0759214027  # kilograms per mole, molar mass of 76Ge

# lines to exclude are
# 2614.511(10) - 511 = 2103.511 keV SEP from 208Tl
# 2118.513(25) keV from 214Bi
# 2204.10(4) keV from 214Bi

# default analysis window (in keV)
WINDOW = [[1930.0, 2098.511], [2108.511, 2113.513], [2123.513, 2190.0]]

# MJD analysis window (in keV) is slightly larger than GERDA/LEGEND and excludes an additional line
MJD_WINDOW = [[1950.0, 2098.511], [2108.511, 2113.513], [2123.513, 2199.1], [2209.1, 2350.0]]

# could use these to go a little faster?
LOG2 = 0.69314718055994528622676398299518041312694549560546875
SQRT2PI = 2.506628274631000241612355239340104162693023681640625

# conversion function
def s_prime_to_s(s_prime):
  # Given s_prime in decays/(kg*yr), find s in decays/yr
  s = s_prime * (M76 / (LOG2 * NA) )
  return s

def s_prime_to_halflife(s_prime):
  # Given s_prime in decays/(kg*yr), find t_half in yrs
  t_half = 1/(s_prime_to_s(s_prime))
  return t_half
