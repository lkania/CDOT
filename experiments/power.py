import time

start_time = time.time()


def runtime(string):
	print(
		"-- {0}: {1} hours --".format(
			string,
			round((time.time() - start_time) / 3600, 2)))


######################################################################
# Utilities
######################################################################
from tqdm import tqdm
from functools import partial
import hashlib

#######################################################
# activate parallelism in CPU for JAX
# See:
# - https://github.com/google/jax/issues/3534
# - https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
#######################################################
import os

import multiprocessing

match multiprocessing.cpu_count():
	case 32:
		n_jobs = 25
	case 12:
		n_jobs = 10

# n_jobs = 20
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
	n_jobs)

#######################################################
# Load Jax
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, clear_backends, random

# def clear():
# 	clear_caches()
# 	clear_backends()


######################################################################
# local libraries
######################################################################
import localize
from src.background.transform import exponential
from src.dotdic import DotDic
from experiments.parser import parse
from experiments import plot, selection, plots
from experiments.builder import load_background_and_signal
from src.test.test import build as build_test
from src import bin
from src.basis import bernstein
from experiments.parallel import run

uniform_bin = lambda X, lower, upper, n_bins: bin.full_uniform_bin(
	n_bins=n_bins)

##################################################
# Simulation parameters
##################################################
args = parse()
args.seed = 0
args.key = random.key(seed=args.seed)
args.dtype = np.float64
args.n_jobs = n_jobs
args.target_data_id = args.data_id
args.use_cache = True
args.alpha = 0.05
args.sample_size = 20000
args.folds = 1000
args.signal_region = [0.1, 0.9]  # the signal region contains 80% of the signal
args.ks = [5, 10, 15, 20, 25, 30, 35, 40, 45]
args.classifiers = ['class', 'tclass']
# Subsets are used for plotting while
# the full range is used for power curve computations
args.lambdas_subset = [0, 0.05]
args.lambdas = [0, 0.01, 0.02, 0.05]

args.quantiles_subset = [0.0, 0.5, 0.7]
args.quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# the following is a number lower than any possible score
# predicted by any classifier
args.zero_cut = 0

assert args.folds > 1
assert args.n_jobs > 1
assert args.folds % args.n_jobs == 0
assert max([len(args.lambdas),
			len(args.quantiles),
			len(args.ks)]) <= len(plot.colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(args.target_data_id)

params = load_background_and_signal(args)


##################################################
# Check that using the zero_cut for any classifier produces
# the original dataset. That is, nothing is filtered.
##################################################
def check_no_filtering(args, params):
	for classifier in args.classifiers:
		d1 = (params.background.X, params.background.c[classifier])
		d2 = (params.signal.X, params.signal.c[classifier])
		for d in [d1, d2]:
			X_ = params.filter(X_=d, cutoff=args.zero_cut)
			assert X_.shape[0] == d[0].shape[0]
			assert np.all((X_ - d[0]) == 0)


check_no_filtering(args, params)


###################################################
# Define test statistic for each classifier x cutoff combination
###################################################

def get_trans(args, min_):
	match args.target_data_id:
		case '3b' | '4b':
			return lambda X: exponential.trans(X=X,
											   rate=0.003,
											   base=min_,
											   scale=1)
		case 'WTagging':
			# In the WTagging dataset, the data is in the [0,1] scale.
			# Hence, no transformation in required.
			return lambda X: X

		case _:
			raise ValueError('Dataset not supported')


# Produce a dataset, filter it, transform it and test it


def generate_dataset(cutoff,
					 classifier,
					 params,
					 sample_size,
					 lambda_,
					 key,
					 trans):
	X, mask = params.subsample_and_mask(n=sample_size,
										classifier=classifier,
										lambda_=lambda_,
										key=key,
										cutoff=cutoff)
	return np.stack((trans(X), mask), axis=0)


selected = DotDic()
for classifier in args.classifiers:
	selected[classifier] = DotDic()
	for quantile in args.quantiles:
		selected[quantile] = None

all = DotDic()
for k in args.ks:
	all[k] = selected.copy()

for classifier in args.classifiers:
	print('Configure {0} classifer'.format(classifier))

	fig, axs = plot.plt.subplots(nrows=1,
								 ncols=len(args.quantiles),
								 figsize=(10 * len(args.quantiles), 10),
								 sharex='all',
								 sharey='all')

	cutoffs = np.quantile(params.background.c[classifier],
						  q=np.array(args.quantiles),
						  axis=0)
	cutoffs = cutoffs.at[0].set(args.zero_cut)

	for q, quantile in enumerate(args.quantiles):

		test = DotDic()
		test.name = '{0}_{1}'.format(classifier, quantile)

		# filtering parameter
		test.cutoff = cutoffs[q]

		# Note: modify here to change the data transformation after thresholding
		# Note that if you do that, there is a high change of
		# test data being below the threshold
		# specify data transformation
		test.trans = get_trans(
			args=args,
			min_=int(np.floor(np.min(params.background.X)))
		)

		test.args = DotDic()
		test.args.bins = 100
		test.args.method = 'bin_mle'
		test.args.optimizer = 'dagostini'
		test.args.fixpoint = 'normal'
		test.args.maxiter = 5000
		test.args.tol = 1e-10
		test.args.alpha = args.alpha

		# Note: modify here to set signal region after thresholding
		qs = np.quantile(params.signal.X,
						 q=np.array(args.signal_region),
						 axis=0)
		test.args.lower = test.trans(qs[0])
		test.args.upper = test.trans(qs[1])

		# Note: uniform binning fails with asymptotic test
		# test.args.from_, test.args.to_ = bin.uniform_bin(
		# 	lower=test.args.lower,
		# 	upper=test.args.upper,
		# 	n_bins=test.args.bins)

		# TODO: switch to equal frequency binning
		# Code for doing equal-frequency bins
		# Get expected number of bins to the
		# left and right of the signal region
		# print('Estimating average equal-frequency binning')
		fs, ts = [], []

		for r in tqdm(range(args.folds), ncols=40):
			X_ = test.trans(
				params.subsample_and_filter(
					n=args.sample_size,
					classifier=classifier,
					lambda_=0,
					cutoff=selected[classifier][quantile].cutoff)
			)
			bins_lower, bins_upper = bin._adaptive_n_bins(
				X=X_,
				lower=dic.lower,
				upper=dic.upper,
				n_bins=dic.bins)
			fs.append(bins_lower)
			ts.append(bins_upper)
		bins_lower = int(np.mean(np.array(fs), axis=0))
		bins_upper = int(np.mean(np.array(ts), axis=0))
		# Get average bins location for the above
		fs, ts = [], []
		for r in tqdm(range(args.folds), ncols=40):
			X_ = selected[classifier][quantile].trans(
				params.subsample_and_filter(
					n=args.sample_size,
					classifier=classifier,
					lambda_=0,
					cutoff=selected[classifier][quantile].cutoff))
			f, t = bin.adaptive_bin(
				X=X_,
				lower=test.args.lower,
				upper=test.args.upper,
				n_bins=(bins_lower, bins_upper))
			fs.append(f)
			ts.append(t)
		test.args.from_ = np.mean(np.array(fs), axis=0)
		test.args.to_ = np.mean(np.array(ts), axis=0)

		# Note: if using equal-frequency bins
		# remove last and first bin for 3b/4b datasets
		match args.target_data_id:
			case '3b' | '4b':
				# remove first and last bin
				test.args.from_ = test.args.from_[1:-1]
				test.args.to_ = test.args.to_[1:-1]

		assert test.args.from_.shape[0] == test.args.to_.shape[0]
		test.args.bins = len(test.args.from_)

		tests = []
		for i, k in enumerate(args.ks):
			tests.append(test.copy())
			tests[i].args.k = k
			tests[i].name = '{0}_{1}'.format(tests[i].name, k)
			tests[i].args.hash = int(
				hashlib.sha1(tests[i].name.encode("utf-8")).hexdigest(), 16)
			tests[i].args.basis = bernstein
			tests[i].generate_dataset = partial(
				generate_dataset,
				trans=tests[i].trans,
				classifier=classifier,
				cutoff=tests[i].cutoff)
			tests[i].test = build_test(args=tests[i].args)

			all[k][classifier][quantile] = tests[i]

		###################################################
		# Path for saving results
		###################################################
		path = '{0}/val/{1}/{2}'.format(args.target_data_id,
										classifier,
										quantile)

		selection_results = selection.select(
			args=args,
			path=path,
			params=params,
			tests=tests,
			measure=partial(selection.target_alpha_level, alpha=args.alpha))

		selected[classifier][quantile] = selection_results.test_star
		del selection_results['test_star']

		# update threshold

		# Note: to fix the polynomial complexity
		# simply uncomment the following code
		# match args.target_data_id:
		# 	case '3b' | '4b':
		# 		# choose k=20
		# 		selected[classifier][quantile] = tests[0]
		# 	case 'WTagging':
		# 		# choose k=30
		# 		selected[classifier][quantile] = tests[1]

		# clear()
		print('Selected Classifier: {0}'.format(
			selected[classifier][quantile].name))

		###################################################
		# Plot fits for different polynomial complexities
		###################################################
		plots.fits(args=args,
				   results=selection_results,
				   path=path,
				   filename='fits_uniform',
				   binning=uniform_bin,
				   alpha=args.alpha)

		plots.fits(args=args,
				   results=selection_results,
				   path=path,
				   filename='fits',
				   alpha=args.alpha)

		# Prepare data
		labels = []
		pvalues = []
		measure = []
		for i, k in enumerate(selection_results.keys()):
			labels.append(k)
			if k == selected[classifier][quantile].name:
				labels[i] += ' (*)'
			pvalues.append(selection_results[k].pvalues)
			measure.append(selection_results[k].measure)

		###################################################
		# Plot all CDF curves
		###################################################
		fig, ax = plot.plt.subplots(nrows=1, ncols=1,
									figsize=(10, 10),
									sharex='none',
									sharey='none')

		ax.set_title('P-value CDF')
		plot.cdfs(ax=ax,
				  df=pvalues,
				  labels=labels,
				  alpha=args.alpha)
		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='pvalues')

		ax.set_xlim([0, args.alpha])
		ax.set_ylim([0, args.alpha])
		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='pvalues_restricted')

		###################################################
		# Plot CI for I(pvalue <= alpha)
		###################################################
		fig, ax = plot.plt.subplots(nrows=1, ncols=1,
									figsize=(10, 10),
									sharex='none',
									sharey='none')
		ax.set_title('Clopper-Pearson CI for I(pvalue<={0})'.format(
			args.alpha))
		ax.axhline(y=args.alpha,
				   color='black',
				   linestyle='-',
				   label='{0}'.format(args.alpha))
		plot.binary_series_with_uncertainty(
			ax,
			x=args.ks,
			values=pvalues,
			color='red',
			alpha=args.alpha)
		ax.legend()
		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='level')

		###################################################
		# Plot measure
		###################################################
		fig, ax = plot.plt.subplots(nrows=1, ncols=1,
									figsize=(10, 10),
									sharex='none',
									sharey='none')
		ax.set_title('Selection measure')
		ax.plot(labels,
				measure,
				color='black')
		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='measure')


def empirical_power(args, path, params, classifier, selected, quantiles,
					lambdas):
	results = DotDic()
	for lambda_ in lambdas:

		results[lambda_] = DotDic()

		for quantile in quantiles:
			print("\nClassifier={0} lambda={1} cutoff={2}\n".format(
				classifier,
				lambda_,
				quantile))

			results[lambda_][quantile] = run(
				args=args,
				params=params,
				test=selected[classifier][quantile],
				path='{0}/storage/{1}/{2}'.format(path, lambda_, quantile),
				lambda_=lambda_)

	# clear()

	return results


def power_analysis(args, params, selected, storage_string, plot_string):
	print("\n{0} power analysis\n".format(storage_string))

	plot_path = '{0}/{1}/{2}'.format(args.target_data_id,
									 storage_string,
									 plot_string)

	results = DotDic()
	for classifier in params.classifiers:
		path = '{0}/{1}/{2}'.format(args.target_data_id,
									storage_string,
									classifier)

		results[classifier] = empirical_power(args=args,
											  path=path,
											  params=params,
											  classifier=classifier,
											  selected=selected,
											  quantiles=args.quantiles,
											  lambdas=args.lambdas)

		plots.filtering(args=args,
						lambdas=args.lambdas_subset,
						quantiles=args.quantiles_subset,
						results=results[classifier],
						path=plot_path,
						filename='{0}_filter'.format(classifier),
						alpha=args.alpha)

		plots.filtering(args=args,
						lambdas=args.lambdas_subset,
						quantiles=args.quantiles_subset,
						results=results[classifier],
						path=plot_path,
						binning=uniform_bin,
						filename='{0}_filter_uniform'.format(classifier),
						alpha=args.alpha)

	plots.power_per_classifier(args=args,
							   path=plot_path,
							   results=results,
							   lambdas=args.lambdas,
							   quantiles=args.quantiles,
							   classifiers=args.classifiers,
							   labels=['Without Decorrelation',
									   'With Decorrelation'])

	plots.power_per_quantile(args=args,
							 path=plot_path,
							 results=results,
							 lambdas=args.lambdas,
							 quantiles=args.quantiles_subset)


# clear()


runtime('Finished model selection')
##################################################
# Validation data analysis
##################################################
power_analysis(args=args,
			   params=params,
			   selected=selected,
			   storage_string='val',
			   plot_string='selected')

for k in all.keys():
	power_analysis(args=args,
				   params=params,
				   selected=all[k],
				   storage_string='val',
				   plot_string='{0}'.format(k))

runtime('Finished power analysis on validation data')
##################################################
# Test data analysis
##################################################
args.data_id = args.target_data_id
params = load_background_and_signal(args)

check_no_filtering(args, params)

power_analysis(args=args,
			   params=params,
			   selected=selected,
			   storage_string='test',
			   plot_string='selected')

for k in all.keys():
	power_analysis(args=args,
				   params=params,
				   selected=all[k],
				   storage_string='test',
				   plot_string='{0}'.format(k))

###################################################################
# Total time
###################################################################
runtime('runtime')
