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

	parser.add_argument(
		'--ks',
		nargs='+',
		help='Order range for model selection',
		default=[1, 2],
		type=int)

	parser.add_argument(
		'--bins_selection',
		help='Percentage of bins used for model selection',
		default=20,
		type=float)

	parser.add_argument("--bins",
						default=100,
						type=int)

	parser.add_argument(
		"--model_signal",
		help='The signal is modelled by a normal distribution with unknown parameters',
		default=False,
		type=lambda x: True if x == 'True' or x == 'true' else False)

	parser.add_argument(
		"--debias",
		help='Split the sample and de-bias',
		default=False,
		type=lambda x: True if x == 'True' or x == 'true' else False)

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
						default='WTagging',
						type=str)

	args, _ = parser.parse_known_args()

	args.classifiers = ["tclass", "class"]

	return args
