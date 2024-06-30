from functools import partial

import numpy as onp
from jax import numpy as np, pmap
from tqdm import tqdm

from experiments import key_management
from experiments import storage
from src.dotdic import DotDic


def split_leading_axis(arr, n_jobs):
	return arr.reshape(arr.shape[0] // n_jobs,
					   n_jobs,
					   *arr.shape[1:])


def generate_datasets(args, keys, test, params, lambda_):
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


def generate_counts(args, keys, test, params, lambda_):
	keys = split_leading_axis(keys, n_jobs=args.n_jobs)
	l = lambda k: test.generate_counts(
		params=params,
		sample_size=args.sample_size,
		lambda_=lambda_,
		key=k)

	exec = pmap(l, in_axes=(0))
	data = onp.zeros(shape=(args.folds, args.sample_size),
					 dtype=args.float)
	masks = onp.zeros(shape=(args.folds, args.sample_size),
					  dtype=onp.int32)
	counts = onp.zeros(shape=(args.folds, test.args.bins),
					   dtype=args.int)
	sample_sizes = onp.zeros(shape=(args.folds),
							 dtype=onp.int32)

	for j in tqdm(range(args.folds // args.n_jobs), ncols=40):
		start = (j * args.n_jobs)
		end = (j + 1) * args.n_jobs
		X, mask, count, n = exec(keys[j])

		assert not np.isnan(X).any()
		assert not np.isnan(mask).any()
		assert not np.isnan(count).any()
		assert np.all(count >= 0)
		assert not np.isnan(n).any()
		assert np.all(n > 0)

		data[start:end, :] = X
		masks[start:end, :] = mask
		counts[start:end, :] = count
		sample_sizes[start:end] = n

	counts = np.array(counts, dtype=args.float)
	sample_sizes = np.array(sample_sizes)

	return data, masks, counts, sample_sizes


def predict_counts(predict_density, k, n, gamma, lambda_, from_, to_):
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
	ppredict_counts = lambda n, gamma, lambda_, from_, to_: predict_counts(
		n=n,
		gamma=gamma,
		lambda_=lambda_,
		from_=from_,
		to_=to_,
		predict_density=predict_density,
		k=k)
	exec = pmap(ppredict_counts,
				in_axes=(0, 0, 0, None, None))

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
	keys = key_management.keys(args=args, num=args.folds)

	if args.use_cache and storage.exists(cwd=args.cwd,
										 path=path,
										 name=test.name):
		data, masks, results_ = storage.load_obj(cwd=args.cwd,
												 path=path,
												 name=test.name)
	else:
		print(
			'\nGenerate datasets in parallel lambda={0}\n'.format(lambda_))
		# data, masks, counts, sample_sizes = generate_counts(
		# 	args=args,
		# 	keys=keys,
		# 	test=test,
		# 	params=params,
		# 	lambda_=lambda_)

		data, mask = generate_datasets(
			args=args,
			keys=keys,
			test=test,
			params=params,
			lambda_=lambda_)

		# Certify that all observations fall between 0 and 1
		# since we are using Bernstein polynomials
		assert (np.max(data * mask) <= 1) and (np.min(data * mask) >= 0)

		# l = lambda c, n: test.test(counts=c, n=n)
		# counts = split_leading_axis(counts, n_jobs=args.n_jobs)
		# sample_sizes = split_leading_axis(sample_sizes.reshape(-1, 1),
		# 								  n_jobs=args.n_jobs)

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
			# mask filtered observations
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

	results.predict_counts = partial(
		parallel_predict_counts,
		n_jobs=args.n_jobs,
		dtype=args.float,
		predict_density=test.args.basis.predict,
		k=test.args.k,
		n=results.n,
		gamma=results.gammas,
		lambda_=results.lambdas)

	results.predict_counts(from_=test.args.from_, to_=test.args.to_)

	return results
