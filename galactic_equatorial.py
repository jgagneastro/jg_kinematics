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

def galactic_equatorial(gl,gb):
	"""Transforms Galactic coordinates (gl,gb) to equatorial coordinates (ra,dec). All inputs must be numpy arrays of the same dimension
	
		param gl: Galactic longitude (degrees)
		param gb: Galactic latitude (degrees)
		output (ra,dec): Tuple containing right ascension and declination (degrees)
	"""
	
	#Check for parameter consistency
	num_stars = np.size(gl)
	if np.size(gb) != num_stars:
		raise ValueError('The dimensions gl and gb do not agree. They must all be numpy arrays of the same length.')
		
	#Compute intermediate quantities
	sin_gb = np.sin(np.radians(gb))
	cos_gb = np.cos(np.radians(gb))
	l_north_m_gl = l_north - gl
	cos_gl = np.cos(np.radians(l_north_m_gl))
	sin_gl = np.sin(np.radians(l_north_m_gl))
	
	#Compute declination
	sin_dec = sin_gb*sin_dec_pol + cos_gb*cos_dec_pol*cos_gl
	dec = np.degrees(np.asin(sin_dec))
	
	#Compute right ascension
	cos_dec = np.sqrt(1.0-sin_dec**2)
	sin_f = cos_gb*sin_gl/cos_dec
	cos_f = (sin_gb - sin_dec_pol*sin_dec)/(cos_dec_pol*cos_dec)
	ra = ra_pol + np.degrees(np.atan2(sin_f,cos_f))
	ra = np.mod(ra,360.0)
	
	#Return equatorial coordinates tuple
	return (ra, dec)