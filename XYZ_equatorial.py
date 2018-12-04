import numpy as np #Numpy maths

def XYZ_equatorial(X, Y, Z, X_error=None, Y_error=None, Z_error=None):
	"""
	Transforms Galactic position XYZ to equatorial coordinates (ra,dec) and distance. All inputs must be numpy arrays of the same dimension.
	
	param X: Galactic position X toward Galactic center (parsec)
	param Y: Galactic position Y in the driection of Galactic motion (parsec)
	param Z: Galactic position Z outside and perpendicular to Galacic plane (toward Galactic North pole; parsec)
	
	output (ra,dec,dist): Tuple containing equatorial position and distance (right ascension in degrees; declination in degrees; distance in parsec)
	output (ra,dec,dist,edist): Tuple containing equatorial position, distance, and measurement error on distance (right ascension in degrees; declination in degrees; distance in parsec, error in parsec), used if any measurement errors are given as input.
	"""
	
	#Verify keywords
	num_stars = np.size(X)
	if np.size(Y) != num_stars or np.size(Z) != num_stars:
		raise ValueError('X, Y and Z must all be numpy arrays of the same size !')
	if (X_error is not None and np.size(X_error) != num_stars) or (Y_error is not None and np.size(Y_error) != num_stars) or (Z_error is not None and np.size(Z_error) != num_stars):
		raise ValueError('X_error, Y_error and Z_error must be numpy arrays of the same size as X, Y and Z !')
	
	#Compute distance
	dist = np.SQRT(X**2 + Y**2 + Z**2)
	
	#Compute Galactic coordinates
	gl = np.degrees(np.arctan2(Y, X))
	XY_dist = np.sqrt(X**2 + Y**2)
	gb = 90.0 - np.degrees(np.arctan2(XY_dist, Z))
	
	#Transform Galactic coordinates to equatorial cooridinates
	(ra, dec) = galactic_equatorial(gl, gb)
	
	#Propagate measurement errors on distance
	if X_error is not None:
		edist = np.SQRT((X*X_error)**2 + (Y*Y_error)**2 + (Z*Z_error)**2)/dist
		return (ra, dec, dist, edist)
	else:
		return (ra, dec, dist)