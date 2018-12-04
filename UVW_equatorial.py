import numpy as np #Numpy maths

#Initiate some global constants
#1 AU/yr to km/s divided by 1000
kappa = 0.004743717361
#Not using "from astropy import units as u; kappa=u.au.to(u.km)/u.year.to(u.s)" because astropy defines one year as exactly 365.25 days instead of 365 days

#Galactic Coordinates matrix
TGAL = (np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
	[0.4941094279, -0.4448296300, 0.7469822445],
	[-0.8676661490,  -0.1980763734, 0.4559837762]]))

def UVW_equatorial(U, V, W, ra, dec, U_error=None, V_error=None, W_error=None, dist=None, dist_error=None):
	"""
	Transforms Space velocities UVW (kilometers per second) and equatorial coordinates (ra, dec) to proper motion (pmra, pmdec), radial velocity and distance. All inputs must be numpy arrays of the same dimension.
	
	param U: Space velocity U toward the Galactic center (kilometers per second)
	param V: Space velocity V toward the direction of Galactic disk motion (kilometers per second)
	param W: Space velocity W perpendicular from the Galactic disk toward Galactic North pole (kilometers per second)
	param ra: Right ascension (degrees)
	param dec: Declination (degrees)
	param U_error: Error on space velocity U toward the Galactic center (kilometers per second)
	param V_error: Error on space velocity V toward the direction of Galactic disk motion (kilometers per second)
	param W_error: Error on space velocity W perpendicular from the Galactic disk toward Galactic North pole 
	param dist: Distance in parsec (facultative)
	param dist_error: Distance error in parsec (facultative)
	
	
	output (reduced_pmra, reduced_pmdec, rv, dist): Tuple containing reduced proper motion (proper motion in right ascension pmra including the cos(declination) term in milliarcsecond per year; reduced proper motion in declination pmdec in milliarcsecond per year), radial velocity (in kilometers per second) and distance (in parsec) if distance is not given.
	
	output (reduced_pmra, reduced_pmdec, rv, dist, reduced_epmra, reduced_epmdec, erv, edist): Tuple containing the same quantities plus their error bars (same units), used if any measurement errors are given as inputs and distance is not given
	
	output (pmra, pmdec, rv, dist): Tuple containing proper motion (proper motion in right ascension pmra including the cos(declination) term in milliarcsecond per year; proper motion in declination pmdec in milliarcsecond per year), radial velocity (in kilometers per second) and distance (in parsec) if distance is given
	
	output (pmra, pmdec, rv, dist, epmra, epmdec, erv, edist): Tuple containing the same quantities plus their error bars (same units), used if any measurement errors are given as inputs and distance is given
	"""
	
	#Verify keywords
	num_stars = np.size(U)
	if np.size(V) != num_stars or np.size(W) != num_stars or np.size(ra) != num_stars or np.size(dec) != num_stars:
		raise ValueError('U, V, W, ra and dec must all be numpy arrays of the same size !')
	if (U_error is not None and np.size(U_error) != num_stars) or (V_error is not None and np.size(V_error) != num_stars) or (W_error is not None and np.size(W_error) != num_stars):
		raise ValueError('U_error, V_error and W_error must be numpy arrays of the same size as U, V, W, ra and dec !')
	
	#Compute elements of the T matrix
	cos_ra = np.cos(np.radians(ra))
	cos_dec = np.cos(np.radians(dec))
	sin_ra = np.sin(np.radians(ra))
	sin_dec = np.sin(np.radians(dec))
	T1 = TGAL[0, 0]*cos_ra*cos_dec + TGAL[0, 1]*sin_ra*cos_dec + TGAL[0, 2]*sin_dec
	T2 = -TGAL[0, 0]*sin_ra + TGAL[0, 1]*cos_ra
	T3 = -TGAL[0, 0]*cos_ra*sin_dec - TGAL[0, 1]*sin_ra*sin_dec + TGAL[0, 2]*cos_dec
	T4 = TGAL[1, 0]*cos_ra*cos_dec + TGAL[1, 1]*sin_ra*cos_dec + TGAL[1, 2]*sin_dec
	T5 = -TGAL[1, 0]*sin_ra + TGAL[1, 1]*cos_ra
	T6 = -TGAL[1, 0]*cos_ra*sin_dec - TGAL[1, 1]*sin_ra*sin_dec + TGAL[1, 2]*cos_dec
	T7 = TGAL[2, 0]*cos_ra*cos_dec + TGAL[2, 1]*sin_ra*cos_dec + TGAL[2, 2]*sin_dec
	T8 = -TGAL[2, 0]*sin_ra + TGAL[2, 1]*cos_ra
	T9 = -TGAL[2, 0]*cos_ra*sin_dec - TGAL[2, 1]*sin_ra*sin_dec + TGAL[2, 2]*cos_dec
	
	#Calculate analytical derivatives to propagate error bars
	if U_error is not None:
		T1_ra = T2*cos_dec
		T2_ra = -TGAL[0, 0]*cos_ra - TGAL[0, 1]*sin_ra
		T3_ra = TGAL[0, 0]*sin_ra*sin_dec - TGAL[0,1]*cos_ra*sin_dec
		T4_ra = T5*cos_dec
		T5_ra = -TGAL[1, 0]*cos_ra - TGAL[1, 1]*sin_ra
		T6_ra = TGAL[1, 0]*sin_ra*sin_dec - TGAL[1,1]*cos_ra*sin_dec
		T7_ra = T8*cos_dec
		T8_ra = -TGAL[2,0]*cos_ra - TGAL[2,1]*sin_ra
		T9_ra = TGAL[2,0]*sin_ra*sin_dec - TGAL[2,1]*cos_ra*sin_dec
		T1_dec = T3
		T2_dec = 0.0
		T3_dec = -T1
		T4_dec = T6
		T5_dec = 0.0
		T6_dec = -T4
		T7_dec = T9
		T8_dec = 0.0
		T9_dec = -T7
		#This above needs to be simplified (see equatorial_UVW.py)
	
	#Calculate observables
	rv = T1*U + T4*V + T7*W
	reduced_pmra = T2*U + T5*V + T8*W
	reduced_pmdec = T3*U + T6*V + T9*W
	
	#If distance is given compute true proper motions
	if dist is not None:
		output_pmra = reduced_pmra/(kappa*dist)
		output_pmdec = reduced_pmdec/(kappa*dist)
	else:
		output_pmra = reduced_pmra
		output_pmdec = reduced_pmdec
	
	if U_error is not None:
		#This needs to be simplified (see equatorial_UVW.py)
		raise ValueError('This part is not coded yet !')
	else:
		#Return measurement
		return (output_pmra, output_pmdec, rv)
	