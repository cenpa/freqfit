QBB = 2039.061  # literature value of available energy in 76Ge double beta decay TODO: citation 10.1103/PhysRevC.81.032501
NA = 6.0221408e23  # Avogadro's number
MA = 0.0759214027  # kilograms per mole, molar mass of 76Ge

# default analysis window (in keV)
WINDOW = [[1930.0, 2099.0], [2109.0, 2114.0], [2124.0, 2190.0]]

# default analysis window (in keV) but only for background region
# used for bkg_region_0vbb
# includes the default analysis window except Qbb +- 25 keV
WINDOW_BKG_REGION_0VBB = [[1930.0, 2014.0], [2064.0, 2099.0], [2109.0, 2114.0], [2124.0, 2190.0]]

# default analysis window (in keV) but only for signal region
# used for sig_region_0vbb
# includes only Qbb +- 25 keV
WINDOW_SIG_REGION_0VBB = [[2014.0, 2064.0]]

# could use these to go a little faster?
LOG2 = 0.69314718055994528622676398299518041312694549560546875
SQRT2PI = 2.506628274631000241612355239340104162693023681640625

# conversion function
def s_prime_to_s(s_prime):
  # Given s_prime in decays/(kg*yr), find s in decays/yr
  s = s_prime * (MA / (LOG2 * NA) )
  return s

def s_prime_to_halflife(s_prime):
  # Given s_prime in decays/(kg*yr), find t_half in yrs
  t_half = 1/(s_prime_to_s)
  return t_half
