from argparse import ArgumentParser


def parse():
	parser = ArgumentParser()

	parser.add_argument("--cwd", type=str, default='..')

	parser.add_argument("--data_id",
						default='3b',
						type=str)

	args, _ = parser.parse_known_args()

	return args
