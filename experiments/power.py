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
from experiments import plot
from experiments import storage
from experiments.builder import load_background, load_signal
from src.test.test import test as _test
from src import bin

uniform_bin = lambda X, lower, upper, n_bins: bin.full_uniform_bin(
	n_bins=n_bins)


def fits(args,
		 params,
		 ks,
		 path,
		 filename,
		 alpha,
		 binning=None):
	n_cols = len(ks)
	fig, axs = plot.plt.subplots(nrows=2,
								 ncols=n_cols,
								 figsize=(10 * len(ks),
										  10),
								 height_ratios=[2, 1],
								 sharex='all',
								 sharey='row')

	axs[0, 0].set_ylabel('Counts ($\lambda=0$)')
	axs[1, 0].set_ylabel('Obs / Pred')

	for i, k in enumerate(ks):
		ax = axs[0, i]
		ax2 = axs[1, i]
		ax2.set_xlabel('K={0}'.format(k))
		test_args.k = k

		tests = []
		for r in range(args.repeat):
			tests.append(
				test(args=test_args,
					 X=params.subsample(n=args.sample_size, lambda_=0)))

		plot.hists(ax,
				   binning=binning,
				   ax2=ax2,
				   methods=tests,
				   alpha=alpha)

	fig = plot.tight_pairs(n_cols=n_cols, fig=fig)
	plot.save_fig(cwd=args.cwd,
				  path=path,
				  fig=fig,
				  name=filename)


def filtering(args,
			  params,
			  classifier,
			  lambdas,
			  quantiles,
			  selected,
			  path,
			  alpha,
			  filename,
			  binning=None,
			  eps=1e-2):
	##################################################
	# Plot datasets with classifier filter
	##################################################
	n_cols = len(quantiles)
	fig, axs = plot.plt.subplots(nrows=len(lambdas) * 2,
								 ncols=n_cols,
								 height_ratios=np.array(
									 [[2, 1]] * len(lambdas)).reshape(-1),
								 figsize=(10 * len(quantiles),
										  10 * len(lambdas)),
								 sharex='all',
								 sharey='row')

	# fig.suptitle(
	# 'Uniform binning (One instance) for {0}'.format(classifier),
	# 			 fontsize=30)

	l = 0
	for lambda_ in lambdas:

		for q, quantile in enumerate(quantiles):

			ax = axs[l, q]
			ax2 = axs[l + 1, q]
			ax.set_xlim([0 - eps, 1 + eps])
			ax2.set_xlim([0 - eps, 1 + eps])

			if l == 0:
				ax.set_title(
					'Filtering {0}% of the observations'.format(
						int(quantile * 100)))
			if q == 0:
				ax.set_ylabel('Counts ($\lambda={0}$)'.format(lambda_))
				ax2.set_ylabel('Obs / Pred')
			if l == len(lambdas):
				ax2.set_xlabel('Mass (projected scale)')

			tests = []
			for r in range(args.repeat):
				tests.append(
					test(args=selected[classifier][quantile].args,
						 X=params.subsample_and_filter(
							 n=args.sample_size,
							 classifier=classifier,
							 lambda_=lambda_,
							 cutoff=selected[classifier][quantile].cutoff)))

			plot.hists(ax,
					   binning=binning,
					   methods=tests,
					   ax2=ax2,
					   alpha=alpha)

		l += 2

	fig = plot.tight_pairs(n_cols=n_cols, fig=fig)
	plot.save_fig(cwd=args.cwd, path=path,
				  fig=fig,
				  name=filename)


def power(ax, results, lambdas, quantiles, alpha, eps=1e-2):
	ax.set_title('Clopper-Pearson CI for I(Test=1) at alpha={0}'.format(alpha))
	ax.set_xlabel('Background reject percentage (BR%)')
	ax.set_ylabel('Empirical probability of rejecting $\lambda=0$')
	ax.set_ylim([0 - eps, 1 + eps])

	ax.axhline(y=alpha,
			   color='black',
			   linestyle='-',
			   label='{0}'.format(alpha))

	for i, lambda_ in enumerate(lambdas):
		plot.binary_series_with_uncertainty(
			ax,
			x=np.array(quantiles) * 100,
			values=[
				np.array(results[lambda_][quantile] <= alpha, dtype=np.int32)
				for quantile in quantiles],
			color=plot.colors[i],
			label='$\lambda$={0}'.format(lambda_),
			alpha=alpha)

	ax.legend()


def power_per_classifier(path, classifiers, labels, results, lambdas,
						 quantiles):
	fig, axs = plot.plt.subplots(nrows=1, ncols=2,
								 figsize=(20, 10),
								 sharex='none',
								 sharey='none')

	for c, classifier in enumerate(classifiers):
		power(ax=axs[c],
			  results=results[classifier],
			  lambdas=lambdas,
			  quantiles=quantiles,
			  alpha=args.alpha)
		ax.set_title(labels[c], fontsize=20)

	plot.save_fig(cwd=args.cwd, path=path, fig=fig, name='power')


def power_per_quantile(path, results, lambdas, quantiles):
	plot_ = lambda classifier, quantile: plot.cdfs(
		ax=ax,
		df=[results[classifier][lambda_][quantile] for lambda_ in
			lambdas],
		labels=['$\lambda=${0}'.format(lambda_) for lambda_ in lambdas])

	for quantile in quantiles:
		fig, axs = plot.plt.subplots(nrows=1, ncols=2,
									 figsize=(20, 10),
									 sharex='none',
									 sharey='none')
		fig.suptitle('CDF of pvalues (BR%={0})'.format(quantile * 100),
					 fontsize=30)

		ax = axs[0]
		ax.set_title('Without Decorrelation', fontsize=30)
		ax.set_title(
			'CDF of pvalues (BR%={0})'.format(quantile * 100))
		plot_('class', quantile)

		ax = axs[1]
		ax.set_title('With Decorrelation', fontsize=30)
		ax.set_title(
			'CDF of pvalues (BR%={0})'.format(quantile * 100))
		plot_('tclass', quantile)

		plot.save_fig(cwd=args.cwd,
					  path=path,
					  fig=fig,
					  name='CDF_for_quantile_{0}'.format(quantile))


# %%
# TODO: unify plotting and experimentation
# that is, try to
##################################################
# Simulation parameters
##################################################
args = parse()
args.target_data_id = args.data_id
args.use_cache = True
args.alpha = 0.05
args.sample_size = 20000
args.folds = 500
args.repeat = 2
args.signal_region = [0.1, 0.9]
args.lambdas = [0, 0.01, 0.02, 0.05]
args.quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
args.ks = [5, 10, 15, 20, 25, 30, 35]
args.classifiers = ['class', 'tclass']

test_args = DotDic()
test_args.bins = 100
# test_args.binning = lambda X, lower, upper, n_bins: bin.adaptive_bin(
# 	X=X,
# 	lower=lower,
# 	upper=upper,
# 	n_bins=n_bins)
test_args.method = 'bin_mle'
test_args.optimizer = 'dagostini'
test_args.fixpoint = 'normal'
test_args.maxiter = 5000
test_args.tol = 1e-6
test_args.alpha = args.alpha

assert max(len(args.lambdas), len(args.quantiles)) <= len(plot.colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(args.target_data_id)

params = load_background(args)
params = load_signal(params)

##################################################
# Set lower and upper arguments for signal region
# based on the true mean and standard deviation
# of the signal
##################################################
qs = np.quantile(
	params.signal.X,
	q=np.array(args.signal_region),
	axis=0)
test_args.lower = qs[0]
test_args.upper = qs[1]

# Print amount of data outside signal region
prop, _ = bin.proportions(
	params.signal.X,
	np.array([test_args.lower]),
	np.array([test_args.upper]))
prop = prop[0]
print(
	'{0:.2f}% of the loaded signal is contained in the signal region'
	.format(prop * 100))


##################################################
# Background transformation parameters
# Our test assumes that data lies between 0 and 1
# hence, you must project the data to that scale
##################################################

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


test_args.trans = get_trans(
	args=args,
	min_=int(np.floor(np.min(params.background.X))))
test_args.lower = test_args.trans(qs[0])
test_args.upper = test_args.trans(qs[1])


def test(args, X):
	return _test(args=args, X=args.trans(X))


##################################################
# Fix binning for test statistic for model selection
##################################################
# test_args.from_, test_args.to_ = test_args.binning(
# 	X=test_args.trans(params.background.X),
# 	lower=test_args.lower,
# 	upper=test_args.upper,
# 	n_bins=test_args.bins)

test_args.from_, test_args.to_ = bin.uniform_bin(
	lower=test_args.lower,
	upper=test_args.upper,
	n_bins=test_args.bins)


##################################################
# Run model selection for base case, i.e. no cutoff
##################################################

def l2_to_uniform(pvalues):
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


def select(args, params, ks, measure):
	results = DotDic()
	for k in ks:
		results[k] = DotDic()

		test_args.k = k
		print("\nValidation K={}\n".format(k))
		pvalues = []
		# TODO: configure test, jit it, and then run in parallel
		for i in tqdm(range(args.folds), ncols=40):
			X = params.subsample(n=args.sample_size, lambda_=0)
			pvalues.append(test(args=test_args, X=X).pvalue)

		# save results
		pvalues = np.array(pvalues)
		results[k].pvalues = pvalues
		results[k].measure = measure(pvalues)

	###################################################
	# Model selection based on p-value distribution
	###################################################
	idx = np.argmin(np.array([results[k].measure for k in ks]))
	results.k_star = ks[idx]

	clear()

	return results


##################################################
# We find the right complexity for our test statistic
# without any sort of signal enriched procedure
# That is, we tune the statistic to be conservative under the null
##################################################

if args.use_cache:
	selection_results = storage.load_obj(cwd=args.cwd,
										 path='{0}/val'.format(
											 args.target_data_id),
										 name='select')
else:
	selection_results = select(args=args,
							   params=params,
							   ks=args.ks,
							   measure=partial(target_alpha_level,
											   alpha=args.alpha))

	storage.save_obj(cwd=args.cwd,
					 path='{0}/val'.format(args.target_data_id),
					 obj=selection_results,
					 name='select')

###################################################
# Define test statistic for each classifier x cutoff combination
###################################################
print("\nSelected K={0}\n".format(selection_results.k_star))
# TODO: change signal region after thresholding
selected = DotDic()
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

	selected[classifier] = DotDic()
	for q, quantile in enumerate(args.quantiles):
		selected[classifier][quantile] = DotDic()

		# filtering parameter
		selected[classifier][quantile].cutoff = cutoffs[q]

		# define test_args for each classifier x quantile
		selected[classifier][quantile].args = DotDic()
		dic = selected[classifier][quantile].args
		dic.update(test_args)
		dic.k = selection_results.k_star

		# Update tranformation to remove some of the zero bins
		dic.trans = get_trans(
			args=args,
			min_=int(np.floor(np.min(params.filter(
				X_=(params.background.X,
					params.background.c[classifier]),
				cutoff=selected[classifier][quantile].cutoff
			)))))

		dic.from_, dic.to_ = bin.uniform_bin(lower=dic.lower,
											 upper=dic.upper,
											 n_bins=dic.bins)

	# Code for doing equal-frequency bins
	# Get expected number of bins to the left and right of the signal region
	# fs, ts = [], []
	# for r in range(args.repeat):
	# 	X_ = dic.trans(params.subsample_and_filter(
	# 		n=args.sample_size,
	# 		classifier=classifier,
	# 		lambda_=0,
	# 		cutoff=selected[classifier][quantile].cutoff))
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
	# for r in range(args.repeat):
	# 	X_ = dic.trans(params.subsample_and_filter(
	# 		n=args.sample_size,
	# 		classifier=classifier,
	# 		lambda_=0,
	# 		cutoff=selected[classifier][quantile].cutoff))
	# 	f, t = adaptive_bin(
	# 		X=X_,
	# 		lower=dic.lower,
	# 		upper=dic.upper,
	# 		n_bins=(bins_lower, bins_upper))
	# 	fs.append(f)
	# 	ts.append(t)
	# dic.from_ = np.mean(np.array(fs), axis=0)
	# dic.to_ = np.mean(np.array(ts), axis=0)

# plot.save_fig(cwd=args.cwd,
# 			  path='{0}/val'.format(args.target_data_id),
# 			  fig=fig,
# 			  name='{0}_binning'.format(classifier))

# TODO: fix can't pickle function
# save_obj(cwd=args.cwd,
# 		 path='{0}/val'.format(args.target_data_id),
# 		 obj=selected,
# 		 name='selected')

###################################################
# Plot fits for different polynomial complexities
###################################################
fits(args=args,
	 params=params,
	 ks=[5, selection_results.k_star, 35],
	 path='{0}/val'.format(args.target_data_id),
	 filename='fits_uniform',
	 binning=uniform_bin,
	 alpha=args.alpha)

fits(args=args,
	 params=params,
	 ks=[5, selection_results.k_star, 35],
	 path='{0}/val'.format(args.target_data_id),
	 filename='fits',
	 alpha=args.alpha)

###################################################
# Plot all CDF curves
###################################################
fig, ax = plot.plt.subplots(nrows=1, ncols=1,
							figsize=(10, 10),
							sharex='none',
							sharey='none')
labels = []
for i, k in enumerate(args.ks):
	labels.append('K={0}'.format(k))
	if k == selection_results.k_star:
		labels[i] += ' (selected)'

ax.set_title('P-value CDF per K'.format(selection_results.k_star))
plot.cdfs(ax=ax,
		  df=[selection_results[k].pvalues for k in args.ks],
		  labels=labels)
plot.save_fig(cwd=args.cwd,
			  path='{0}/val'.format(args.target_data_id),
			  fig=fig,
			  name='pvalues')

ax.set_xlim([0, args.alpha])
plot.save_fig(cwd=args.cwd,
			  path='{0}/val'.format(args.target_data_id),
			  fig=fig,
			  name='pvalues_restricted')

###################################################
# Plot CI for I(pvalue <= alpha)
###################################################
fig, ax = plot.plt.subplots(nrows=1, ncols=1,
							figsize=(10, 10),
							sharex='none',
							sharey='none')
ax.set_title(
	'Clopper-Pearson CI for I(pvalue<={0}). K={1} selected'.format(
		args.alpha,
		selection_results.k_star))
ax.axhline(y=args.alpha,
		   color='black',
		   linestyle='-',
		   label='{0}'.format(args.alpha))
plot.binary_series_with_uncertainty(
	ax,
	x=args.ks,
	values=[selection_results[k].pvalues for k in args.ks],
	color='red',
	alpha=args.alpha)
ax.legend()
plot.save_fig(cwd=args.cwd,
			  path='{0}/val'.format(args.target_data_id),
			  fig=fig,
			  name='level')

###################################################
# Plot measure
###################################################
fig, ax = plot.plt.subplots(nrows=1, ncols=1,
							figsize=(10, 10),
							sharex='none',
							sharey='none')
ax.set_title(
	'Selection measure. K={0} selected'.format(
		selection_results.k_star))
ax.plot(args.ks,
		[selection_results[k].measure for k in args.ks],
		color='black',
		label='Selection measure')
ax.legend()
plot.save_fig(cwd=args.cwd,
			  path='{0}/val'.format(args.target_data_id),
			  fig=fig,
			  name='measure')


# %%
def empirical_power(args, params, classifier, selected, quantiles, lambdas):
	results = DotDic()
	for lambda_ in lambdas:

		results[lambda_] = DotDic()

		for quantile in quantiles:

			results[lambda_][quantile] = []

			# Run procedures on all datasets
			print("\nTest Classifier={0} lambda={1} cutoff={2}\n".format(
				classifier,
				lambda_,
				quantile))

			for i in tqdm(range(args.folds), ncols=40):
				results[lambda_][quantile].append(
					test(args=selected[classifier][quantile].args,
						 X=params.subsample_and_filter(
							 n=args.sample_size,
							 classifier=classifier,
							 lambda_=lambda_,
							 cutoff=selected[classifier][quantile].cutoff)
						 ).pvalue)

			results[lambda_][quantile] = np.array(results[lambda_][quantile])

			clear()

	return results


def power_analysis(params, string):
	path = '{0}/{1}'.format(args.target_data_id, string)

	for classifier in params.classifiers:
		filtering(args=args,
				  params=params,
				  classifier=classifier,
				  lambdas=[0, 0.05],
				  quantiles=[0.0, 0.5, 0.7],
				  selected=selected,
				  path=path,
				  filename='{0}_filter'.format(classifier),
				  alpha=args.alpha)

		filtering(args=args,
				  params=params,
				  classifier=classifier,
				  lambdas=[0, 0.05],
				  quantiles=[0.0, 0.5, 0.7],
				  selected=selected,
				  path=path,
				  binning=uniform_bin,
				  filename='{0}_filter_uniform'.format(classifier),
				  alpha=args.alpha)

	if args.use_cache:
		results = storage.load_obj(cwd=args.cwd,
								   path=path,
								   name='power')
	else:
		results = DotDic()
		for classifier in params.classifiers:
			results[classifier] = empirical_power(args=args,
												  params=params,
												  classifier=classifier,
												  selected=selected,
												  quantiles=args.quantiles,
												  lambdas=args.lambdas)
		storage.save_obj(cwd=args.cwd,
						 path=path,
						 obj=results,
						 name='power')

	power_per_classifier(
		path=path,
		results=results,
		lambdas=args.lambdas,
		quantiles=args.quantiles,
		classifiers=args.classifiers,
		labels=['Without Decorrelation',
				'With Decorrelation'])

	power_per_quantile(path=path,
					   results=results,
					   lambdas=args.lambdas,
					   quantiles=[0, 0.5, 0.7])

	clear()


##################################################
# Validation data analysis
##################################################
power_analysis(params=params, string='val')
##################################################
# Test data analysis
##################################################
args.data_id = args.target_data_id
params = load_background(args)
params = load_signal(params)
power_analysis(params=params, string='test')

###################################################################
# Total time
###################################################################
print(
	"-- Runtime: %s hours --" % (round((time.time() - start_time) / 3600, 2)))
