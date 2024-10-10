from jax import numpy as np

from experiments.parallel import run
from src.dotdic import DotDic


def l2_to_uniform(args, pvalues):
	inc = (np.arange(args.folds) + 1) / args.folds
	###################################################
	# select K that minimizes the L_2 distance
	# between the empirical CDF of the p-values and
	# uniform CDF
	###################################################
	# Let F be the CDF of the pvalues
	# int_0^1 (F(x)-x)^2 dx = A + B + C
	# where A = int_0^1 F(x)^2 dx
	# B = int_0^1 x^2 dx
	# C = - 2 int_0^1 F(x) x dx
	# For the purpose of minimizing A + B + C w.r.t. K
	# the B term can be ignored.
	# Remember that F(x) = n^{-1} \sum_{i=1}^n I(p_i <= x)
	# Hence, let p_1 <= p_2 <= ... <= p_n <= p_{n+1} = 1
	pvalues = np.sort(pvalues)
	ext_pvalues = np.concatenate((pvalues[1:], np.array([1])))
	# A = \sum_{i=1}^n  (i/n) (p_{i+1} - p_i)
	pvalues_diff = ext_pvalues - pvalues
	A = np.sum(np.square(inc) * pvalues_diff)
	# C = - 2 int_0^1 F(x) x dx
	# C = - 2 \sum_{i=1}^{n} (i/n) \int_{p_i}^{p_{i+1}} x dx
	# C = - \sum_{i=1}^{n} (i/n) [(p_{i+1})^2-(p_{i})^2]
	pvalues_diff = np.square(ext_pvalues) - np.square(pvalues)
	C = - np.sum(inc * pvalues_diff)
	return A + C


def target_alpha_level(pvalues, alpha):
	return np.abs(np.quantile(pvalues, q=alpha) - alpha)


def select(args, path, params, tests, measure):
	path = '{0}/storage'.format(path)
	results = DotDic()
	for test in tests:
		print("\nValidation Classifier: {0}\n".format(test.name))

		results[test.name] = run(args=args,
								 params=params,
								 test=test,
								 path=path,
								 lambda_=0)

		results[test.name].measure = measure(results[test.name].stats)

	###################################################
	# Model selection based on p-value distribution
	###################################################
	idx = np.argmin(np.array([results[test.name].measure for test in tests]))
	results.test_star = tests[idx]

	return results
