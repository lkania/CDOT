from functools import partial

import numpy as onp
from jax import numpy as np, pmap
from tqdm import tqdm

from experiments import key_management
from experiments import storage
from src.dotdic import DotDic


def split_leading_axis(arr, n_jobs):
	"""Group an array's leading dimension into batches for parallel jobs.
	
	Args:
	    arr (array): Input with shape (N, ...), where N is divisible by n_jobs.
	    n_jobs (int): Parallel batch size; a scalar.
	Returns:
	    array: Reshaped input with shape (N / n_jobs, n_jobs, ...).
	"""
	return arr.reshape(arr.shape[0] // n_jobs,
					   n_jobs,
					   *arr.shape[1:])


def generate_datasets(args, keys, test, params, lambda_):
	"""Generate all simulated datasets and selection masks in parallel.
	
	Args:
	    args (Namespace/DotDic): Scalar simulation configuration; uses folds, sample_size, and n_jobs.
	    keys (array): PRNG-key batch with leading dimension args.folds.
	    test (DotDic): Scalar test object exposing generate_dataset.
	    params (DotDic): Scalar data/sampling configuration object.
	    lambda_ (float): Signal fraction; a scalar.
	Returns:
	    tuple: Data and integer masks, each with shape (args.folds, args.sample_size).
	"""
	keys = split_leading_axis(keys, n_jobs=args.n_jobs)
	l = lambda k: test.generate_dataset(
		params=params,
		sample_size=args.sample_size,
		lambda_=lambda_,
		key=k)

	exec = pmap(l, in_axes=(0))
	data = onp.zeros(shape=(args.folds, args.sample_size), dtype=args.float)
	masks = onp.zeros(shape=(args.folds, args.sample_size),
					  dtype=onp.int32)
	# sample_sizes = onp.zeros(shape=(args.folds), dtype=onp.int32)

	for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
		start = (j * args.n_jobs)
		end = (j + 1) * args.n_jobs
		X, mask, n = exec(keys[j])

		assert not np.isnan(X).any()
		assert not np.isnan(mask).any()
		assert not np.isnan(n).any()
		assert np.all(n > 0)

		data[start:end, :] = X
		masks[start:end, :] = mask
	# sample_sizes[start:end] = n

	# sample_sizes = np.array(sample_sizes)

	return data, masks  # , sample_sizes


def predict_counts(predict_density, k, n, gamma, lambda_, from_, to_):
	"""Convert a fitted background density into expected bin counts.
	
	Args:
	    predict_density (callable): Function returning bin probabilities from gamma and bin bounds.
	    k (int): Bernstein polynomial order; a scalar.
	    n (int/float): Number of selected events; a scalar.
	    gamma (array): Background coefficients with shape (k + 1,).
	    lambda_ (float): Estimated signal fraction; a scalar.
	    from_ (array): Lower bin edges with shape (B,).
	    to_ (array): Upper bin edges with shape (B,).
	Returns:
	    array: Expected background counts with shape (B,).
	"""
	density = predict_density(
		gamma=gamma,
		k=k,
		from_=from_,
		to_=to_).reshape(-1)
	return density * n * (1 - lambda_)


def parallel_predict_counts(n_jobs,
							dtype,
							predict_density,
							k,
							n,
							gamma,
							lambda_,
							from_,
							to_):
	"""Compute expected background counts for all simulation folds in parallel.
	
	Args:
	    n_jobs (int): Number of parallel jobs; a scalar.
	    dtype (dtype): NumPy/JAX numeric dtype; a scalar dtype object.
	    predict_density (callable): Function returning bin probabilities.
	    k (int): Bernstein polynomial order; a scalar.
	    n (array): Selected-event counts with shape (F,).
	    gamma (array): Background coefficients with shape (F, k + 1).
	    lambda_ (array): Signal estimates with shape (F,).
	    from_ (array): Lower bin edges with shape (B,).
	    to_ (array): Upper bin edges with shape (B,).
	Returns:
	    ndarray: Expected counts with shape (F, B).
	"""
	ppredict_counts = lambda n, gamma, lambda_, from_, to_: predict_counts(
		n=n,
		gamma=gamma,
		lambda_=lambda_,
		from_=from_,
		to_=to_,
		predict_density=predict_density,
		k=k)
	exec = pmap(ppredict_counts, in_axes=(0, 0, 0, None, None))

	folds = n.shape[0]
	results = onp.zeros(shape=(folds, from_.shape[0]), dtype=dtype)
	n = split_leading_axis(n, n_jobs=n_jobs)
	gamma = split_leading_axis(gamma, n_jobs=n_jobs)
	lambda_ = split_leading_axis(lambda_, n_jobs=n_jobs)

	print('\nPredict counts in parallel\n')
	for j in tqdm(range(folds // n_jobs), ncols=40):
		start = (j * n_jobs)
		end = (j + 1) * n_jobs
		r = exec(n[j], gamma[j], lambda_[j], from_, to_)
		assert not np.isnan(r).any()
		results[start:end, :] = r

	return results


def run(args, params, test, path, lambda_):
	"""Run or load all folds for one test configuration and organize the results.
	
	Args:
	    args (Namespace/DotDic): Scalar simulation configuration with folds, dtypes, and cache settings.
	    params (DotDic): Scalar data/sampling configuration object.
	    test (DotDic): Scalar configured test object with generate_dataset and test callables.
	    path (str): Relative cache/output path; a scalar string.
	    lambda_ (float): Signal fraction used to generate datasets; a scalar.
	Returns:
	    DotDic: Fold-level results, including stats (F,), lambdas (F,), gammas (F, k+1), and n (F,).
	"""
	keys = key_management.keys(args=args, num=args.folds)

	if args.use_cache and storage.exists(cwd=args.cwd,
										 path=path,
										 name=test.name):
		data, mask, results_ = storage.load_obj(cwd=args.cwd,
												path=path,
												name=test.name)
	else:
		print('\nGenerate datasets in parallel lambda={0}\n'.format(
			lambda_))

		data, mask = generate_datasets(
			args=args,
			keys=keys,
			test=test,
			params=params,
			lambda_=lambda_)

		# Certify that all observations fall between 0 and 1
		# since we are using Bernstein polynomials
		assert (np.max(data * mask) <= 1) and (np.min(data * mask) >= 0)

		l = lambda X, mask: test.test(X=X, mask=mask)
		datas = split_leading_axis(data, n_jobs=args.n_jobs)
		masks = split_leading_axis(mask, n_jobs=args.n_jobs)
		exec = pmap(l, in_axes=(0, 0))

		print('\nCompute tests in parallel\n')
		results_ = []
		for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
			r = exec(datas[j], masks[j])

			for k in r.keys():
				assert not np.isnan(r[k]).any()

			results_.append(r)

		storage.save_obj(cwd=args.cwd,
						 path=path,
						 obj=(data, mask, results_),
						 name=test.name)

	print('\nProcess results\n')
	results = DotDic()
	results.test = test
	results.runs = []
	results.stats = onp.zeros(shape=(args.folds), dtype=args.float)
	results.lambdas = onp.zeros(shape=(args.folds), dtype=args.float)
	results.gammas = onp.zeros(shape=(args.folds, test.args.k + 1),
							   dtype=args.float)
	results.n = onp.zeros(shape=(args.folds), dtype=args.int)
	keys = [k for k in results_[0].keys()]

	assert len(results_) * len(results_[0][keys[0]]) == args.folds

	index = 0
	for i in tqdm(range(len(results_)), ncols=40):
		for j in range(len(results_[0][keys[0]])):
			d = DotDic()
			for k in keys:
				d[k] = results_[i][k][j]
			d.X = data[index].reshape(-1)
			d.mask = mask[index].reshape(-1)
			###########################################
			# Mask filtered observations
			# Assign -1 to observations that have been filtered
			###########################################
			d.X = d.X * d.mask + (-1) * (1 - d.mask)
			d.n = np.sum(d.mask)
			d.test = test
			d.predict_counts = partial(
				predict_counts,
				predict_density=test.args.basis.predict,
				k=test.args.k,
				gamma=d.gamma_hat,
				lambda_=d.lambda_hat,
				n=d.n)
			results.runs.append(d)
			results.stats[index] = d.stat
			results.lambdas[index] = d.lambda_hat
			results.gammas[index, :] = d.gamma_hat.reshape(-1)
			results.n[index] = d.n
			index += 1

	results.stats = np.array(results.stats)
	results.lambdas = np.array(results.lambdas)
	results.n = np.array(results.n)
	results.gammas = np.array(results.gammas)

	assert len(results.runs) == args.folds
	assert len(results.stats) == args.folds
	assert len(results.lambdas) == args.folds
	assert len(results.n) == args.folds
	assert results.gammas.shape[0] == args.folds

	# parallel predict counts over all folds
	results.predict_counts = partial(
		parallel_predict_counts,
		n_jobs=args.n_jobs,
		dtype=args.float,
		predict_density=test.args.basis.predict,
		k=test.args.k,
		n=results.n,
		gamma=results.gammas,
		lambda_=results.lambdas)

	return results
