from jax import numpy as np, clear_caches, clear_backends, pmap, jit, random
from tqdm import tqdm
from experiments import storage
from functools import partial
from src.bin import proportions
import numpy as onp
from src.dotdic import DotDic


def split_leading_axis(arr, n_jobs):
	return arr.reshape(arr.shape[0] // n_jobs,
					   n_jobs,
					   *arr.shape[1:])


def run(args, params, test, path, lambda_):
	# generate keys
	keys = random.split(args.key, num=args.folds + 1)
	args.key = keys[-1]

	if args.use_cache and storage.exists(cwd=args.cwd,
										 path=path,
										 name=test.name):
		data, results_ = storage.load_obj(cwd=args.cwd,
										  path=path,
										  name=test.name)
	else:
		# dim(sum) = folds x n_bins x 1
		print('Generating datasets in parallel')
		# TODO: speed up dataset generation by compiling also
		# the computation of proportions. You can easily mask
		# the thresholded observations by assigning them a negative value
		# (any value outside [0,1] will work)
		# TODO: assert that all generated data is in [0,1]
		keys = keys[:-1]
		keys = split_leading_axis(keys, n_jobs=args.n_jobs)
		l = lambda k: test.generate_dataset(
			params=params,
			sample_size=args.sample_size,
			lambda_=lambda_,
			key=k)

		# TODO: You could cache a pmap for each lambda
		# you should check how to still enable params
		# probably you should have a static argum for the
		# functions needed from params
		exec = pmap(l, in_axes=(0))
		data_and_mask = onp.zeros(
			shape=(args.folds, 2, args.sample_size),
			dtype=args.dtype)

		for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
			data_and_mask[(j * args.n_jobs):  ((j + 1) * args.n_jobs), :,
			:] = exec(keys[j])

		print('Filtering datasets sequentially')
		# mask datasets
		data = []
		sum = onp.zeros(shape=(args.folds, test.args.bins + 1),
						dtype=args.dtype)
		for j in tqdm(range(args.folds), ncols=40):
			X = data_and_mask[j][0, :].reshape(-1)
			mask = np.array(data_and_mask[j][1, :], dtype=np.bool_).reshape(-1)
			X_masked = X[mask]

			data.append(X_masked)
			# TODO: parallelize proportions
			sum[j, :-1] = proportions(
				X=X_masked,
				from_=test.args.from_,
				to_=test.args.to_)[0]
			sum[j, -1] = X_masked.shape[0]
			assert not np.isnan(sum[j]).any()

		l = lambda X: test.test(X=X)
		sum = np.array(sum)
		sum = split_leading_axis(sum, n_jobs=args.n_jobs)
		exec = pmap(l, in_axes=(0))

		print('Compute tests in parallel')
		results_ = []
		for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
			r = exec(sum[j])

			# check for NaNs
			for k in r.keys():
				assert not np.isnan(r[k]).any()

			results_.append(r)

		storage.save_obj(cwd=args.cwd,
						 path=path,
						 obj=(data, results_),
						 name=test.name)

	print('Process results')
	results = DotDic()
	results.test = test
	results.runs = []
	results.pvalues = []
	keys = results_[0].keys()
	for i in tqdm(range(len(results_)), ncols=40):
		for j in range(len(results_[0])):
			d = DotDic()
			for k in keys:
				d[k] = results_[i][k][j]
			d.X = data[i + j]
			d.test = test
			d.predict = partial(
				test.args.basis.predict,
				gamma=d.gamma_hat,
				k=test.args.k)
			results.runs.append(d)
			results.pvalues.append(d.pvalue)

	results.pvalues = np.array(results.pvalues)

	return results
