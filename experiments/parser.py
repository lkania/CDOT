from argparse import ArgumentParser


def parse():
	parser = ArgumentParser()

	######################################################################
	# Method parameters
	######################################################################

	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--method",
						type=str,
						default='bin_mle')

	parser.add_argument("--optimizer",
						type=str,
						default='dagostini')

	parser.add_argument("--fixpoint",
						type=str,
						default='normal')

	parser.add_argument("--maxiter",
						type=int,
						default=100)

	parser.add_argument("--tol",
						type=float,
						default=1e-6)

	parser.add_argument(
		"--rate", default=None, type=float)
	parser.add_argument(
		"--a", default=None, type=float)
	parser.add_argument(
		'--b',
		default=None,
		type=lambda x: None if x == 'None' else float(x))
	parser.add_argument(
		"--k",
		help='Order of basis',
		default=None,
		type=lambda x: None if x == 'None' else int(x))

	parser.add_argument("--bins",
						default=100,
						type=int)

	######################################################################
	# Simulation parameters
	######################################################################
	parser.add_argument("--cwd", type=str, default='..')

	parser.add_argument("--folds",
						default=3,
						type=int)

	parser.add_argument("--sample_size",
						default=15000,
						type=int)

	parser.add_argument("--data_id",
						default='3b',
						type=str)

	args, _ = parser.parse_known_args()

	args.classifiers = ["tclass", "class"]

	return args
