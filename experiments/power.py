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
from functools import partial

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
	case 4:
		n_jobs = 2
	case _:
		n_jobs = int(multiprocessing.cpu_count()) / 2

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
	n_jobs)

#######################################################
# Load Jax
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, jit, random as _random
import numpy as onp

######################################################################
# local libraries
######################################################################
import localize
from src.background.transform import exponential
from src.dotdic import DotDic
from experiments.parser import parse
from experiments import plot, selection, plots
from experiments.builder import load_background_and_signal
from src.test.builder import build as build_test
from src import bin
from src.basis import bernstein, normalized_bernstein
from experiments import parallel
from experiments import key_management
import hasher

uniform_bin = lambda X, lower, upper, n_bins: bin.full_uniform_bin(
	n_bins=n_bins)

##################################################
# Simulation parameters
##################################################
args = parse()
args.seed = 0
args.random = _random
args.key = key_management.init(args=args)
args.dtype = np.float64
args.float = np.float64
args.int = np.int64
args.tol = 1e-12
args.target_data_id = args.data_id
args.use_cache = True
args.alpha = 0.05
args.sample_size = 20000
args.folds = 1000
args.n_jobs = min(args.folds, n_jobs)
args.signal_region = [0.1, 0.90]  # the signal region quantiles
args.ks = [5, 10, 15, 20, 25, 30, 35, 40]
args.classifiers = ['class', 'tclass']
args.classifiers_labels = ['Without Decorrelation', 'With Decorrelation']

# Subsets are used for plotting while
# the full range is used for power curve computations
args.lambdas_subset = [0, 0.05]
args.lambdas = [0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

args.quantiles_subset = [0.0, 0.5, 0.7]
args.quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# args.zero_cut is a number lower than any possible classifier score
args.zero_cut = 0

# a flag for model selection
runonce = False

assert np.all(np.array(args.lambdas[1:]) > args.tol)
assert args.folds > 1
assert args.n_jobs > 1
assert args.folds % args.n_jobs == 0
assert max([len(args.lambdas),
			len(args.quantiles),
			len(args.ks)]) <= len(plot.colors)


##################################################
# Compute amount of signal leaking into the control region
##################################################
def __stat(signal_region_start,
		   signal_region_stop,
		   background,
		   signal):
	signal_region_start = np.array([signal_region_start])
	signal_region_stop = np.array([signal_region_stop])
	signal_on_signal_region = bin.counts(
		signal,
		from_=signal_region_start,
		to_=signal_region_stop)[0] / signal.shape[0]
	background_on_control_region = 1 - bin.counts(
		background,
		from_=signal_region_start,
		to_=signal_region_stop)[0] / background.shape[0]
	print('\t % of signal on signal region {0}\n'
		  '\t % of background on control region {1}\n'
		  '\t signal on control / background on control={2}\n'.format(
		signal_on_signal_region,
		background_on_control_region,
		(1 - signal_on_signal_region) / background_on_control_region))


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
			assert np.all(d[1] >= args.zero_cut)


print('Validation dataset:\n ')
args.data_id = '{0}/val'.format(args.target_data_id)
params_ = load_background_and_signal(args)

qs = np.quantile(params_.signal.X,
				 q=np.array(args.signal_region),
				 axis=0)

__stat(signal_region_start=qs[0],
	   signal_region_stop=qs[1],
	   background=params_.background.X,
	   signal=params_.signal.X)

check_no_filtering(args, params_)

print('Test dataset:\n ')
args.data_id = args.target_data_id
params_ = load_background_and_signal(args)

__stat(signal_region_start=qs[0],
	   signal_region_stop=qs[1],
	   background=params_.background.X,
	   signal=params_.signal.X)

check_no_filtering(args, params_)

del params_
del __stat

print('Starting simulation:\n ')
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(args.target_data_id)
params = load_background_and_signal(args)


###################################################
# Define test statistic for each classifier x cutoff combination
###################################################

# Produce a dataset, filter it, and transform it
@partial(jit, static_argnames=['cutoff',
							   'trans',
							   'classifier',
							   'params',
							   'sample_size',
							   'lambda_'])
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
	X = X.reshape(-1)
	mask = np.array(mask.reshape(-1), dtype=np.int32)
	n = np.sum(mask)
	return trans(X), mask, n


all = DotDic()
for k in args.ks:
	all[k] = DotDic()
	for classifier in args.classifiers:
		all[k][classifier] = DotDic()
		for quantile in args.quantiles:
			all[k][classifier][quantile] = None

for classifier in args.classifiers:
	print('Configure {0} classifer'.format(classifier))

	cutoffs = onp.quantile(params.background.c[classifier],
						   q=args.quantiles,
						   axis=0)
	cutoffs[0] = args.zero_cut

	for q, quantile in enumerate(args.quantiles):

		test = DotDic()
		test.name = '{0}_{1}'.format(classifier, quantile)

		# filtering parameter
		test.cutoff = cutoffs[q]

		# Note: modify here to change the data transformation after thresholding
		# Currently, the data transformation does not depend on the
		# threshold cutoff
		match args.target_data_id:
			case '3b' | '4b':
				base_ = int(np.floor(np.min(params.background.X)))
				test.trans = lambda X: exponential.trans(
					X=X,
					rate=0.003,
					base=base_,
					scale=1)
			case 'WTagging':
				# In the WTagging dataset, the data is in the [0,1] scale.
				# Hence, no transformation in required.
				test.trans = lambda X: X

			case _:
				raise ValueError('Dataset not supported')

		test.args = DotDic()
		test.args.bins = 500
		test.args.method = 'bin_mle'
		test.args.optimizer = 'normalized_dagostini'
		test.args.fixpoint = 'normal'
		test.args.maxiter = 5000
		test.args.tol = args.tol
		test.args.alpha = args.alpha
		test.args.float = np.float64
		test.args.int = np.int64

		# Note: modify here to set signal region after thresholding
		# Currently, the signal region does not depend on the cutoff threshold
		qs = np.quantile(params.signal.X,
						 q=np.array(args.signal_region),
						 axis=0)
		test.args.lower = test.trans(qs[0])
		test.args.upper = test.trans(qs[1])

		# Uniform binning
		test.args.from_, test.args.to_ = bin.uniform_bin(
			lower=test.args.lower,
			upper=test.args.upper,
			n_bins=test.args.bins)

		assert test.args.from_.shape[0] == test.args.to_.shape[0]
		test.args.bins = len(test.args.from_)

		tests = []
		for i, k in enumerate(args.ks):
			tests.append(test.copy())
			tests[i].args.k = k
			tests[i].name = '{0}_{1}'.format(tests[i].name, k)
			tests[i].args.hash = hasher.hash_(tests[i].name)
			tests[i].args.basis = normalized_bernstein
			tests[i].generate_dataset = partial(
				generate_dataset,
				trans=tests[i].trans,
				classifier=classifier,
				cutoff=tests[i].cutoff)

			print('Configure {0} test'.format(tests[i].name))

			assert not np.isnan(tests[i].args.from_).any()
			assert not np.isnan(tests[i].args.to_).any()
			assert np.all(tests[i].args.from_ >= 0)
			assert np.all(tests[i].args.to_ >= 0)

			assert tests[i].args.from_.shape[0] == tests[i].args.to_.shape[0]

			tests[i].args.bins = len(tests[i].args.from_)

			tests[i].test = build_test(args=tests[i].args)

			all[k][classifier][quantile] = tests[i]

		###################################################
		# Perform plots and model selection under the null
		# That corresponds to quantile=0, i.e. the classifier
		# output is not used.
		# The following code can be easily modified if it
		# is necessary to perfom model selection for each quantile
		###################################################

		if quantile == 0 and runonce == False:
			runonce = True

			fig, axs = plot.plt.subplots(nrows=1,
										 ncols=len(args.quantiles),
										 figsize=(10 * len(args.quantiles), 10),
										 sharex='all',
										 sharey='all')

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
				measure=partial(selection.target_alpha_level,
								alpha=args.alpha))

			selected_test = selection_results.test_star
			del selection_results['test_star']

			print('Selected Classifier: {0}'.format(selected_test.name))

			# Prepare data
			labels = []
			stats = []
			measure = []
			for i, k in enumerate(selection_results.keys()):
				labels.append('K={0}'.format(selection_results[k].test.args.k))
				if k == selected_test.name:
					labels[i] += ' (*)'
				stats.append(selection_results[k].stats)
				measure.append(selection_results[k].measure)

			###################################################
			# Plot pvalue distribution
			###################################################
			fig, axs = plot.plt.subplots(nrows=1, ncols=2,
										 figsize=(20, 10),
										 sharex='none',
										 sharey='none')

			ax = axs[0]
			ax.set_title('P-value CDF')
			plot.cdfs(ax=ax,
					  df=stats,
					  labels=labels,
					  alpha=args.alpha)
			ax.legend().set_visible(False)

			ax = axs[1]
			ax.set_title('P-value CDF')
			plot.cdfs(ax=ax,
					  df=stats,
					  labels=labels,
					  alpha=args.alpha)
			ax.set_xlim([0, 2 * args.alpha])
			ax.set_ylim([0, 2 * args.alpha])
			ax.legend(loc='lower right')

			plot.save_fig(cwd=args.cwd,
						  path=path,
						  fig=fig,
						  name='pvalues')

			###################################################
			# Plot CI for I(pvalue <= alpha)
			###################################################
			fig, ax = plot.plt.subplots(nrows=1,
										ncols=1,
										figsize=(10, 10),
										sharex='none',
										sharey='none')

			ax.axhline(y=args.alpha,
					   color='red',
					   linestyle='--',
					   label=r'Target Type I error $\alpha=${0}'.format(
						   args.alpha))
			ax.axvline(x=selected_test.args.k,
					   color='blue',
					   linestyle='--',
					   label=r'$K_*$={0}'.format(selected_test.args.k))

			idx = [np.int32(np.array(stat) <= args.alpha) for stat in stats]
			plot.binary_series_with_uncertainty(
				ax,
				x=args.ks,
				values=idx,
				label='CP CI for Type I error'.format(
					args.alpha),
				color='black',
				alpha=args.alpha)

			ax.set_xlabel(r'Polynomial order $K$')
			ax.set_ylabel(r'Probability of rejecting $\lambda=0$')
			ax.legend()
			plot.save_fig(cwd=args.cwd,
						  path=path,
						  fig=fig,
						  name='selection')

###################################################
# Sanity check
# Assert that for the zero quantile, any classifier produces exactly the
# same dataset if the same key is used
###################################################
for k in args.ks:
	for lambda_ in args.lambdas:
		d1 = all[k][args.classifiers[0]][0.0].generate_dataset(
			params=params,
			sample_size=args.sample_size,
			lambda_=lambda_,
			key=args.key)[0]
		d2 = all[k][args.classifiers[1]][0.0].generate_dataset(
			params=params,
			sample_size=args.sample_size,
			lambda_=lambda_,
			key=args.key)[0]
		assert np.all(d1 - d2 == 0)


###################################################
# Plot measure
###################################################
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

			results[lambda_][quantile] = parallel.run(
				args=args,
				params=params,
				test=selected[classifier][quantile],
				path='{0}/storage/{1}/{2}'.format(path, lambda_, quantile),
				lambda_=lambda_)

			# Here we assume we have a pvalue
			results[lambda_][quantile].tests = np.array(
				results[lambda_][quantile].stats <= args.alpha,
				dtype=np.int32)

	return results


def power_analysis(args, params, selected, plot_string):
	print("\n{0} power analysis\n".format(args.data_id))

	plot_path = '{0}/{1}'.format(args.data_id, plot_string)

	results = DotDic()
	for classifier in params.classifiers:
		path = '{0}/{1}'.format(args.data_id, classifier)

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
						binning=uniform_bin,
						filename='{0}_filter_uniform'.format(classifier),
						alpha=args.alpha)

	plots.power_per_classifier(args=args,
							   path=plot_path,
							   results=results,
							   lambdas=args.lambdas,
							   quantiles=args.quantiles,
							   classifiers=args.classifiers,
							   labels=args.classifiers_labels)


runtime('Finished model selection')

##################################################
# Power analysis on validation data
# Used to
##################################################

# Do power-analysis for selected test
power_analysis(args=args,
			   params=params,
			   selected=all[selected_test.args.k],
			   plot_string='{0}'.format(selected_test.args.k))

##################################################
# Power analysis on test data
##################################################
args.data_id = '{0}/test'.format(args.target_data_id)
params = load_background_and_signal(args)

# Do power-analysis for selected test
power_analysis(args=args,
			   params=params,
			   selected=all[selected_test.args.k],
			   plot_string='{0}'.format(selected_test.args.k))

###################################################################
# Total runtime
###################################################################
runtime('runtime')
