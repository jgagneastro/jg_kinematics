import numpy as np #Numpy maths

def equatorial_XYZ(ra, dec, dist, dist_error=None):
	"""
	Transforms equatorial coordinates (ra, dec) and distance to Galactic position XYZ. All inputs must be numpy arrays of the same dimension.
	
	param ra: Right ascension (degrees)
	param dec: Declination (degrees)
	param dist: Distance (parsec)
	param dist_error: Error on distance (parsec)
	
	output (X, Y, Z): Tuple containing Galactic position XYZ (parsec)
	output (X, Y, Z, EX, EY, EZ): Tuple containing Galactic position XYZ and their measurement errors, used if any measurement errors are given as inputs (parsec)
	"""
	
	#Verify keywords
	num_stars = np.size(ra)
	if np.size(dec) != num_stars or np.size(dist) != num_stars:
		raise ValueError('ra, dec and distance must all be numpy arrays of the same size !')
	if dist_error is not None and np.size(dist_error) != num_stars:
		raise ValueError('dist_error must be a numpy array of the same size as ra !')
	
	#Compute Galactic coordinates
	(gl, gb) = equatorial_galactic(ra, dec)
	
	cos_gl = np.cos(np.radians(gl))
	cos_gb = np.cos(np.radians(gb))
	sin_gl = np.sin(np.radians(gl))
	sin_gb = np.sin(np.radians(gb))
	
	X = cos_gb * cos_gl * dist
	Y = cos_gb * sin_gl * dist
	Z = sin_gb * dist
	
	if dist_error is not None:
		#X_gb = sin_gb * cos_gl * dist * np.pi/180.
		#X_gl = cos_gb * sin_gl * dist * np.pi/180.
		X_dist = cos_gb * cos_gl
		EX = np.abs(X_dist * dist_error)
		Y_dist = cos_gb * sin_gl
		EY = np.abs(Y_dist * dist_error)
		Z_dist = sin_gb
		EZ = np.abs(Z_dist * dist_error)
		return (X, Y, Z, EX, EY, EZ)
	else:
		return (X, Y, Z)