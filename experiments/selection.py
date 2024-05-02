from jax import numpy as np, clear_caches, clear_backends, pmap
from src.dotdic import DotDic
from tqdm import tqdm
from experiments import storage
from functools import partial


def clear():
	clear_caches()
	clear_backends()


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
	return np.abs(alpha - np.mean(pvalues <= alpha))


def run(args, params, test, path):
	if args.use_cache and storage.exists(cwd=args.cwd,
										 path=path,
										 name=test.name):
		results_ = storage.load_obj(cwd=args.cwd,
									path=path,
									name=test.name)
	else:

		print('Prepare data')
		data = []
		for j in tqdm(range(args.folds), ncols=40):
			data.append(test.generate_dataset(
				params=params,
				sample_size=args.sample_size,
				lambda_=0))
		data = np.array(data)
		data_ = data.reshape(data.shape[0] // args.n_jobs,
							 args.n_jobs,
							 *data.shape[1:])

		print('Parallelize tests')
		results_ = []
		for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
			r = pmap(lambda X: test.test(X=X), in_axes=(0))(data_[j])

			# check for NaNs
			for k in r.keys():
				assert not np.isnan(r[k]).any()

			results_.append(r)

		storage.save_obj(cwd=args.cwd,
						 path=path,
						 obj=results_,
						 name=test.name)

	print('Process results')
	results = DotDic()
	results.test = test
	results.runs = []
	results.pvalues = []
	keys = results_[0].keys()
	for i in tqdm(range(args.folds // args.n_jobs), ncols=40):
		for j in range(args.n_jobs):
			d = DotDic()
			for k in keys:
				d[k] = results_[i][k][j]
			d.test = test
			d.predict = partial(
				test.args.basis.predict,
				gamma=d.gamma_hat,
				k=test.args.k)
			results.runs.append(d)
			results.pvalues.append(d.pvalue)

	results.pvalues = np.array(results.pvalues)

	return results


def select(args, path, params, tests, measure):
	path = '{0}/storage'.format(path)
	results = DotDic()
	for test in tests:
		print("\nValidation Classifier: {0}\n".format(test.name))

		results[test.name] = run(args, params, test, path)

		results[test.name].measure = measure(results[test.name].pvalues)

		clear()

	###################################################
	# Model selection based on p-value distribution
	###################################################
	idx = np.argmin(np.array([results[test.name].measure for test in tests]))
	results.test_star = tests[idx]

	return results
