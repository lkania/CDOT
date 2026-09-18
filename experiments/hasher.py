import hashlib


##
# Define hash function
#
def hash_(string):
	"""Convert a string identifier to a deterministic SHA-1 integer hash.
	
	Args:
	    string (str): Input text; a scalar string.
	Returns:
	    int: Scalar integer representation of the SHA-1 digest.
	"""
	return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16)
