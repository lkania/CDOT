def init(args):
	return args.random.key(seed=args.seed)


def keys(args, num):
	keys = args.random.split(args.key, num=num + 1)
	args.key = keys[-1]
	keys = keys[:-1]
	return keys
