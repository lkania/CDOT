from argparse import ArgumentParser


def parse():
	"""Parse experiment command-line options while ignoring unknown arguments.
	
	Returns:
	    argparse.Namespace: Scalar configuration object containing cwd and data_id.
	"""
	parser = ArgumentParser()

	parser.add_argument("--cwd", type=str, default='..')

	parser.add_argument("--data_id",
						default='WTagging',
						type=str)

	args, _ = parser.parse_known_args()

	return args
