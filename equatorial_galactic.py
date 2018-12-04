import numpy as np #Numpy maths

#Initiate some global constants
#J2000.0 Equatorial position of the Galactic North (b=90 degrees) from Carrol and Ostlie
ra_pol = 192.8595
dec_pol = 27.12825

#J2000.0 Galactic latitude gb of the Celestial North pole (dec=90 degrees) from Carrol and Ostlie
l_north = 122.932

#Initiate some secondary variables
sin_dec_pol = np.sin(np.radians(dec_pol))
cos_dec_pol = np.cos(np.radians(dec_pol))

def equatorial_galactic(ra,dec):
	"""Transforms equatorial coordinates (ra,dec) to Galactic coordinates (gl,gb). All inputs must be numpy arrays of the same dimension
	
		param ra: Right ascension (degrees)
		param dec: Declination (degrees)
		output (gl,gb): Tuple containing Galactic longitude and latitude (degrees)
	"""
	
	#Check for parameter consistency
	num_stars = np.size(ra)
	if np.size(dec) != num_stars:
		raise ValueError('The dimensions ra and dec do not agree. They must all be numpy arrays of the same length.')
	
	#Compute intermediate quantities
	ra_m_ra_pol = ra - ra_pol
	sin_ra = np.sin(np.radians(ra_m_ra_pol))
	cos_ra = np.cos(np.radians(ra_m_ra_pol))
	sin_dec = np.sin(np.radians(dec))
	cos_dec = np.cos(np.radians(dec))
	
	#Compute Galactic latitude
	gamma = sin_dec_pol*sin_dec + cos_dec_pol*cos_dec*cos_ra
	gb = np.degrees(np.arcsin(gamma))
	
	#Compute Galactic longitude
	x1 = cos_dec * sin_ra
	x2 = (sin_dec - sin_dec_pol*gamma)/cos_dec_pol
	gl = l_north - np.degrees(np.arctan2(x1,x2))
	gl = np.mod(gl,360.0)
	
	#Return Galactic coordinates tuple
	return (gl, gb)