def init(args):
	"""Initialize the experiment PRNG key from the configured seed.
	
	Args:
	    args (Namespace/DotDic): Scalar configuration object with seed and JAX random module.
	Returns:
	    JAX PRNG key: One initialized random-number-generator key.
	"""
	return args.random.key(seed=args.seed)


def keys(args, num):
	"""Split and advance the stored PRNG key for repeated simulation folds.
	
	Args:
	    args (Namespace/DotDic): Scalar configuration object containing the current PRNG key.
	    num (int): Number of keys requested; a scalar.
	Returns:
	    array: Batch of PRNG keys with leading dimension num.
	"""
	keys = args.random.split(args.key, num=num + 1)
	args.key = keys[-1]
	keys = keys[:-1]
	return keys
