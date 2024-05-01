import time

start_time = time.time()

######################################################################
# Utilities
######################################################################
from tqdm import tqdm
from functools import partial
#######################################################
# activate parallelism in CPU for JAX
# See:
# - https://github.com/google/jax/issues/3534
# - https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
#######################################################
# import os
# import multiprocessing
#
# n_jobs = multiprocessing.cpu_count() - 2
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
# 	n_jobs)

#######################################################
# Load Jax
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, clear_caches, clear_backends


def clear():
	clear_caches()
	clear_backends()


######################################################################
# local libraries
######################################################################
import localize
from src.background.transform import exponential
from src.dotdic import DotDic
from experiments.parser import parse
from experiments import plot, storage, selection, plots
from experiments.builder import load_background, load_signal
from src.test.test import test as _test
from src import bin

uniform_bin = lambda X, lower, upper, n_bins: bin.full_uniform_bin(
	n_bins=n_bins)

##################################################
# Simulation parameters
##################################################
args = parse()
args.target_data_id = args.data_id
args.use_cache = True
args.alpha = 0.05
args.sample_size = 20000
args.folds = 250
args.signal_region = [0.1, 0.9]  # the signal region contains 80% of the signal
args.ks = [5, 10, 15, 20, 25, 30, 35, 40, 45]
args.classifiers = ['class', 'tclass']
# Subsets are used for plotting while
# the full range is used for power curve computations
args.lambdas = [0, 0.01, 0.02, 0.05]
args.lambdas_subset = [0, 0.05]
args.quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
args.quantiles_subset = [0.0, 0.5, 0.7]

assert args.folds > 1

test_args = DotDic()
test_args.bins = 100
test_args.method = 'bin_mle'
test_args.optimizer = 'dagostini'
test_args.fixpoint = 'normal'
test_args.maxiter = 5000
test_args.tol = 1e-10
test_args.alpha = args.alpha

assert max(len(args.lambdas), len(args.quantiles)) <= len(plot.colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(args.target_data_id)

params = load_background(args)
params = load_signal(params)


###################################################
# Define test statistic for each classifier x cutoff combination
###################################################

def get_trans(args, min_):
	match args.target_data_id:
		case '3b' | '4b':
			return lambda X: exponential.safe_trans(X=X,
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
def random_test(args, cutoff, trans, classifier, params, sample_size, lambda_):
	return _test(
		args=args,
		X=trans(
			params.subsample_and_filter(
				n=sample_size,
				classifier=classifier,
				lambda_=lambda_,
				cutoff=cutoff)
		))


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

	for q, quantile in enumerate(args.quantiles):
		selected[classifier][quantile] = DotDic()

		selected[classifier][quantile].name = '{0}_{1}'.format(classifier,
															   quantile)

		# filtering parameter
		selected[classifier][quantile].cutoff = cutoffs[q]

		# Note: modify here to change the data transformation after thresholding
		# Note that if you do that, there is a high change of
		# test data being below the threshold
		# specify data transformation
		selected[classifier][quantile].trans = get_trans(
			args=args,
			min_=int(np.floor(np.min(params.background.X)))
		)

		selected[classifier][quantile].args = test_args.copy()
		dic = selected[classifier][quantile].args

		# Note: modify here to set signal region after thresholding
		qs = np.quantile(params.signal.X,
						 q=np.array(args.signal_region),
						 axis=0)
		dic.lower = selected[classifier][quantile].trans(qs[0])
		dic.upper = selected[classifier][quantile].trans(qs[1])

		# Note: uniform binning fails with asymptotic test
		dic.from_, dic.to_ = bin.uniform_bin(lower=dic.lower,
											 upper=dic.upper,
											 n_bins=dic.bins)

		# Code for doing equal-frequency bins
		# Get expected number of bins to the
		# left and right of the signal region
		# print('Estimating average equal-frequency binning')
		# fs, ts = [], []
		# for r in tqdm(range(args.folds), ncols=40):
		# 	X_ = selected[classifier][quantile].trans(
		# 		params.subsample_and_filter(
		# 			n=args.sample_size,
		# 			classifier=classifier,
		# 			lambda_=0,
		# 			cutoff=selected[classifier][quantile].cutoff)
		# 	)
		# 	bins_lower, bins_upper = bin._adaptive_n_bins(
		# 		X=X_,
		# 		lower=dic.lower,
		# 		upper=dic.upper,
		# 		n_bins=dic.bins)
		# 	fs.append(bins_lower)
		# 	ts.append(bins_upper)
		# bins_lower = int(np.mean(np.array(fs), axis=0))
		# bins_upper = int(np.mean(np.array(ts), axis=0))
		# # Get average bins location for the above
		# fs, ts = [], []
		# for r in tqdm(range(args.folds), ncols=40):
		# 	X_ = selected[classifier][quantile].trans(
		# 		params.subsample_and_filter(
		# 			n=args.sample_size,
		# 			classifier=classifier,
		# 			lambda_=0,
		# 			cutoff=selected[classifier][quantile].cutoff))
		# 	f, t = bin.adaptive_bin(
		# 		X=X_,
		# 		lower=dic.lower,
		# 		upper=dic.upper,
		# 		n_bins=(bins_lower, bins_upper))
		# 	fs.append(f)
		# 	ts.append(t)
		# dic.from_ = np.mean(np.array(fs), axis=0)
		# dic.to_ = np.mean(np.array(ts), axis=0)

		# Note: if using equal-frequency bins
		# remove last and first bin for 3b/4b datasets
		# match args.target_data_id:
		# 	case '3b' | '4b':
		# 		# remove first and last bin
		# 		dic.from_ = dic.from_[1:-1]
		# 		dic.to_ = dic.to_[1:-1]

		selected[classifier][quantile].random_test = partial(
			random_test,
			classifier=classifier,
			args=selected[classifier][quantile].args,
			trans=selected[classifier][quantile].trans,
			cutoff=selected[classifier][quantile].cutoff)

		# TODO: separate building the test from testing,
		# that is, you can ammortize most of the preprocessing steps
		# create a build function that returns an object with test
		# function
		tests = []
		for i, k in enumerate(args.ks):
			tests.append(selected[classifier][quantile].copy())
			tests[i].args.k = k
			tests[i].name = '{0}_{1}'.format(tests[i].name, k)
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
			measure=partial(
				selection.target_alpha_level,
				alpha=args.alpha))

		selected[classifier][quantile] = selection_results.test_star
		del selection_results['test_star']

		# Note: to fix the polynomial complexity
		# simply uncomment the following code
		# match args.target_data_id:
		# 	case '3b' | '4b':
		# 		# choose k=20
		# 		selected[classifier][quantile] = tests[0]
		# 	case 'WTagging':
		# 		# choose k=30
		# 		selected[classifier][quantile] = tests[1]

		clear()
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
				labels[i] += ' (selected)'
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
				  labels=labels)
		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='pvalues')

		ax.set_xlim([0, args.alpha])
		ax.set_ylim([0, 0.2])  # TODO: do this automatically
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

			path_ = '{0}/storage/{1}/{2}'.format(path, lambda_, quantile)
			name_ = selected[classifier][quantile].name
			if args.use_cache and storage.exists(cwd=args.cwd,
												 path=path_,
												 name=name_):
				results[lambda_][quantile] = storage.load_obj(
					cwd=args.cwd,
					path=path_,
					name=name_)
			else:

				results[lambda_][quantile] = DotDic()
				results[lambda_][quantile].pvalues = []
				results[lambda_][quantile].runs = []

				# Run procedures on all datasets
				print("\nClassifier={0} lambda={1} cutoff={2}\n".format(
					classifier,
					lambda_,
					quantile))

				for i in tqdm(range(args.folds), ncols=40):
					t = selected[classifier][quantile].random_test(
						params=params,
						sample_size=args.sample_size,
						lambda_=lambda_
					)
					results[lambda_][quantile].runs.append(t)
					results[lambda_][quantile].pvalues.append(t.pvalue)

				results[lambda_][quantile].pvalues = np.array(
					results[lambda_][quantile].pvalues)

				storage.save_obj(cwd=args.cwd,
								 path=path_,
								 obj=results[lambda_][quantile],
								 name=name_)

				clear()

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

	clear()


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

##################################################
# Test data analysis
##################################################
args.data_id = args.target_data_id
params = load_background(args)
params = load_signal(params)

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
print(
	"-- Runtime: %s hours --" % (round((time.time() - start_time) / 3600, 2)))
