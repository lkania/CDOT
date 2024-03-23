import time

start_time = time.time()

######################################################################
# Configure matplolib
######################################################################
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

config = {'figure.dpi': 600,
		  'legend.fontsize': 'xx-large',
		  'axes.labelsize': 'xx-large',
		  'axes.titlesize': 'xx-large',
		  'xtick.labelsize': 'x-large',
		  'ytick.labelsize': 'x-large'}
pylab.rcParams.update(config)

colors = ['red',
		  'limegreen',
		  'blue',
		  'magenta',
		  'cyan',
		  'darkorange',
		  'grey',
		  'tab:pink',
		  'tab:olive']

######################################################################
# Utilities
######################################################################
from tqdm import tqdm
from pathlib import Path
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
# allow 64 bits
#######################################################
from jax.config import config

config.update("jax_enable_x64", True)

from jax import numpy as np, clear_caches, clear_backends, random


def clear():
	clear_caches()
	clear_backends()


######################################################################
# local libraries
######################################################################
import localize
from src.dotdic import DotDic
from experiments.parser import parse
from experiments.builder import load_background, load_signal
from src.test.test import test as _test
from src.stat.binom import exact_binomial_ci, exact_poisson_ci
from src.bin import proportions, adaptive_bin, full_adaptive_bin, \
	uniform_bin as _uniform_bin

uniform_bin = lambda X, lower, upper, n_bins: _uniform_bin(n_bins=n_bins)


###################################################################
# save figure
###################################################################
def get_path(cwd, path):
	path_ = '{0}/results/{1}/'.format(cwd, path)
	Path(path_).mkdir(parents=True, exist_ok=True)
	return path_


def save_fig(cwd, path, fig, name):
	filename = get_path(cwd=cwd, path=path) + '{0}.pdf'.format(name)
	fig.savefig(fname=filename, bbox_inches='tight')
	plt.close(fig)
	print('\nSaved to {0}'.format(filename))


######################################################################
# plot utilities
######################################################################

# See: https://stackoverflow.com/questions/51717199/how-to-adjust-space-between-every-second-row-of-subplots-in-matplotlib
def tight_pairs(n_cols, fig=None):
	"""
	Stitch vertical pairs together.

	Input:
	- n_cols: number of columns in the figure
	- fig: figure to be modified. If None, the current figure is used.

	Assumptions:
	- fig.axes should be ordered top to bottom (ascending row number).
	  So make sure the subplots have been added in this order.
	- The upper-half's first subplot (column 0) should always be present

	Effect:
	- The spacing between vertical pairs is reduced to zero by moving all lower-half subplots up.

	Returns:
	- Modified fig
	"""
	if fig is None:
		fig = plt.gcf()
	for ax in fig.axes:
		if hasattr(ax, 'get_subplotspec'):
			ss = ax.get_subplotspec()
			row, col = ss.num1 // n_cols, ss.num1 % n_cols
			if (row % 2 == 0) and (col == 0):  # upper-half row (first subplot)
				y0_upper = ss.get_position(fig).y0
			elif (row % 2 == 1):  # lower-half row (all subplots)
				x0_low, _, width_low, height_low = ss.get_position(fig).bounds
				ax.set_position(
					pos=[x0_low, y0_upper - height_low, width_low, height_low])
	return fig


def binary_rv_uncertainty(values, alpha):
	values_ = np.array(values, dtype=np.int32)
	cp = exact_binomial_ci(n_successes=[np.sum(values_)],
						   n_trials=values_.shape[0],
						   alpha=alpha)[0]
	mean = np.mean(values_)
	lower = cp[0]
	upper = cp[1]

	return lower, mean, upper


def plot_series_with_uncertainty(ax, x, mean,
								 lower=None,
								 upper=None,
								 label='',
								 color='black',
								 fmt='',
								 markersize=5,
								 elinewidth=1,
								 capsize=3,
								 set_xticks=True):
	if set_xticks:
		ax.set_xticks(x)

	mean = np.array(mean).reshape(-1)
	yerr = None
	if lower is not None and upper is not None:
		lower = np.array(lower).reshape(-1)
		upper = np.array(upper).reshape(-1)
		l = mean - lower
		u = upper - mean
		yerr = np.vstack((l, u)).reshape(2, -1)

	ax.errorbar(x=x,
				y=mean,
				yerr=yerr,
				color=color,
				capsize=capsize,
				markersize=markersize,
				elinewidth=elinewidth,
				fmt=fmt,
				label=label)


def plot_binary_series_with_uncertainty(ax,
										x,
										values,
										alpha,
										label='',
										color='black',
										fmt='',
										markersize=5,
										elinewidth=1,
										capsize=3,
										set_xticks=True):
	means = []
	lowers = []
	uppers = []
	for v in values:
		lower, mean, upper = binary_rv_uncertainty(values=v, alpha=alpha)
		means.append(mean)
		lowers.append(lower)
		uppers.append(upper)

	return plot_series_with_uncertainty(ax=ax,
										x=x,
										mean=means,
										lower=lowers,
										upper=uppers,
										label=label,
										color=color,
										fmt=fmt,
										markersize=markersize,
										elinewidth=elinewidth,
										capsize=capsize,
										set_xticks=set_xticks)


def plot_hist_with_uncertainty(ax,
							   from_,
							   to_,
							   mean,
							   lower=None,
							   upper=None,
							   jitter=0,
							   color='black',
							   markersize=2,
							   label=''):
	bin_centers = jitter + (from_ + to_) / 2
	plot_series_with_uncertainty(ax=ax,
								 x=bin_centers,
								 mean=mean,
								 lower=lower,
								 upper=upper,
								 color=color,
								 label=label,
								 set_xticks=False,
								 markersize=markersize,
								 elinewidth=1,
								 capsize=1,
								 fmt='o')


def plot_hists(ax,
			   binning,
			   methods,
			   alpha,
			   ax2=None,
			   jitter=1e-1,
			   markersize=5):
	params = methods[0].params
	from_, to_ = binning(
		X=methods[0].X,
		lower=methods[0].params.lower,
		upper=methods[0].params.upper,
		n_bins=methods[0].params.bins)

	props = proportions(X=methods[0].X, from_=from_, to_=to_)[0]
	counts = props * methods[0].X.shape[0]

	# Re-scale Poisson CI so that it's in the same scale as the normalized counts
	cis = exact_poisson_ci(n_events=counts, alpha=alpha) / methods[0].X.shape[0]

	ax.axvline(x=methods[0].params.lower,
			   color='red', linestyle='--')
	ax.axvline(x=methods[0].params.upper,
			   color='red', linestyle='--',
			   label='Signal region')
	if ax2 is not None:
		ax2.axvline(x=methods[0].params.lower,
					color='red', linestyle='--')
		ax2.axvline(x=methods[0].params.upper,
					color='red', linestyle='--',
					label='Signal region')
		ax2.axhline(y=1,
					color='black',
					linestyle='-')

	plot_hist_with_uncertainty(
		ax=ax,
		from_=from_,
		to_=to_,
		mean=props,
		lower=cis[:, 0].reshape(-1),
		upper=cis[:, 1].reshape(-1),
		color='black',
		label='Data (Exact Poisson CI)',
		markersize=2)
	for i, method in enumerate(methods):
		label = 'K={0} p-value={1}'.format(method.k,
										   round(method.pvalue, 2))
		prediction = params.basis.predict(
			gamma=method.gamma_hat,
			k=method.k,
			from_=from_,
			to_=to_).reshape(-1)
		plot_hist_with_uncertainty(
			ax=ax,
			from_=from_,
			to_=to_,
			mean=prediction,
			jitter=(i + 1) * jitter * (to_ - from_),
			color=colors[i],
			markersize=markersize,
			label=label)

		if ax2 is not None:
			plot_hist_with_uncertainty(
				ax=ax2,
				from_=from_,
				to_=to_,
				mean=props / prediction,
				jitter=(i + 1) * jitter * (to_ - from_),
				color=colors[i],
				markersize=markersize - 1,
				label=label)

	ax.legend()


def plot_cdfs(ax, df, labels, eps=1e-2):
	ax.set_ylim([0 - eps, 1 + eps])
	ax.set_xlim([0 - eps, 1 + eps])
	ax.axline([0, 0], [1, 1], color='black', label='Uniform CDF')

	for i, d in enumerate(df):
		sns.ecdfplot(
			data=d,
			hue_norm=(0, 1),
			legend=False,
			ax=ax,
			color=colors[i],
			alpha=1,
			label=labels[i])

	ax.set_ylabel('Cumulative probability')
	ax.set_xlabel('pvalue')
	ax.legend()


def fits(args,
		 params,
		 ks,
		 path,
		 filename,
		 alpha):
	n_cols = len(ks)
	fig, axs = plt.subplots(nrows=2,
							ncols=n_cols,
							figsize=(10 * len(ks),
									 10),
							height_ratios=[2, 1],
							sharex='all',
							sharey='row')

	X = params.subsample(n=args.sample_size, lambda_=0)

	axs[0, 0].set_ylabel('$\lambda=0$ Normalized counts')
	axs[1, 0].set_ylabel('Obs / Pred')

	for i, k in enumerate(ks):
		ax = axs[0, i]
		ax2 = axs[1, i]
		ax2.set_xlabel('K={0}'.format(k))
		test_args.k = k
		test_ = test(args=test_args, X=X)
		plot_hists(ax,
				   uniform_bin,
				   ax2=ax2,
				   methods=[test_],
				   jitter=0,
				   markersize=5,
				   alpha=alpha)

	fig = tight_pairs(n_cols=n_cols, fig=fig)
	save_fig(cwd=args.cwd,
			 path=path,
			 fig=fig,
			 name=filename)


def filtering(args,
			  params,
			  classifier,
			  lambdas,
			  quantiles,
			  path,
			  alpha,
			  filename,
			  k,
			  eps=1e-2):
	assert k is not None

	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(quantiles),
		axis=0)

	##################################################
	# Plot datasets with classifier filter
	##################################################
	n_cols = len(quantiles)
	fig, axs = plt.subplots(nrows=len(lambdas) * 2,
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

		X_ = params.subsample(
			n=args.sample_size,
			classifier=classifier,
			lambda_=lambda_, )

		for c, cutoff in enumerate(cutoffs):

			quantile = quantiles[c]
			X = params.filter(X_=X_, cutoff=cutoff)

			ax = axs[l, c]
			ax2 = axs[l + 1, c]
			ax.set_xlim([0 - eps, 1 + eps])
			ax2.set_xlim([0 - eps, 1 + eps])
			ax2.set_ylim([0 - eps, 4 + eps])

			if l == 0:
				ax.set_title(
					'Filtering {0}% of the observations'.format(
						int(quantile * 100)))
			if c == 0:
				ax.set_ylabel('$\lambda={0}$ Normalized counts'.format(lambda_))
				ax2.set_ylabel('Obs / Pred')
			if l == len(lambdas):
				ax2.set_xlabel('Mass (projected scale)')

			args.k = k
			test_ = test(args=test_args, X=X)
			plot_hists(ax,
					   uniform_bin,
					   methods=[test_],
					   ax2=ax2,
					   jitter=0,
					   markersize=5,
					   alpha=alpha)

		l += 2

	fig = tight_pairs(n_cols=n_cols, fig=fig)
	save_fig(cwd=args.cwd, path=path,
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
		plot_binary_series_with_uncertainty(
			ax,
			x=np.array(quantiles) * 100,
			values=[
				np.array(results[lambda_][quantile] <= alpha, dtype=np.int32)
				for quantile in quantiles],
			color=colors[i],
			label='$\lambda$={0}'.format(lambda_),
			alpha=alpha)

	ax.legend()


def power_per_classifier(path, classifiers, labels, results, lambdas,
						 quantiles):
	fig, axs = plt.subplots(nrows=1, ncols=2,
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

	save_fig(cwd=args.cwd, path=path, fig=fig, name='power')


###################################################################
# save results
###################################################################
import pickle


def save_obj(cwd, path, obj, name):
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('\nSaved to {0}'.format(filename))


def load_obj(cwd, path, name):
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	with open(filename, 'rb') as handle:
		obj = pickle.load(handle)
	return obj


# %%
##################################################
# Simulation parameters
##################################################
args = parse()
args.target_data_id = args.data_id
args.use_cache = False
args.alpha = 0.05
args.sample_size = 20000
args.folds = 500
args.signal_region = [0.1, 0.9]
args.lambdas = lambdas = [0, 0.01, 0.02, 0.05]
args.quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
args.ks = [5, 10, 15, 20, 25, 30, 35]
args.classifiers = ['class', 'tclass']

test_args = DotDic()
test_args.bins = 100
test_args.method = 'bin_mle'
test_args.optimizer = 'dagostini'
test_args.fixpoint = 'normal'
test_args.maxiter = 1000
test_args.tol = 1e-6
test_args.alpha = args.alpha

assert max(len(args.lambdas), len(args.quantiles)) <= len(colors)
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
prop, _ = proportions(
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

match args.target_data_id:
	case '3b' | '4b':
		from src.background.transform import exponential

		a = float(np.round(np.min(params.background.X)))
		test_args.trans = lambda X: exponential.trans(X=X,
													  rate=0.003,
													  base=a,
													  scale=1)

	case 'WTagging':

		test_args.trans = lambda X: X

	case _:
		raise ValueError('Dataset not supported')

test_args.lower = test_args.trans(qs[0])
test_args.upper = test_args.trans(qs[1])


def test(args, X):
	return _test(args=args, X=test_args.trans(X))


##################################################
# Fix binning for test statistic for validation
##################################################
tX = test_args.trans(params.background.X)
test_args.from_, test_args.to_ = adaptive_bin(
	X=tX,
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
	# TODO:
	#  implement plot_hists_with_uncertainty from test.py
	# It will have high memory consumption

	results = DotDic()
	for k in ks:
		results[k] = DotDic()

		test_args.k = k
		print("\nValidation K={}\n".format(k))
		pvalues = []
		# TODO: configure test, jit it, and then run in parallel
		for i in tqdm(range(args.folds), ncols=40):
			X = params.subsample(n=args.sample_size, lambda_=0)
			test_ = test(args=test_args, X=X)
			pvalues.append(test_.pvalue)

		# save results
		pvalues = np.array(pvalues)
		results[k].pvalues = pvalues
		results[k].measure = measure(pvalues)

	###################################################
	# Model selection based on p-value distribution
	###################################################
	idx = np.argmin(np.array([results[k].measure for k in ks]))
	k_star = ks[idx]
	results.k_star = k_star

	clear()

	return results


##################################################
# We find the right complexity for our test statistic
# without any sort of signal enriched procedure
# That is, we tune the statistic to be conservative under the null
##################################################

if args.use_cache:
	results = load_obj(cwd=args.cwd,
					   path='{0}/val'.format(args.target_data_id),
					   name='select')
else:
	results = select(args=args,
					 params=params,
					 ks=args.ks,
					 measure=partial(target_alpha_level, alpha=args.alpha))

	save_obj(cwd=args.cwd,
			 path='{0}/val'.format(args.target_data_id),
			 obj=results,
			 name='select')

test_args.k_star = results.k_star
print("\nSelected K={0}\n".format(test_args.k_star))

###################################################
# Define test statistic for each classifier x cutoff combination
###################################################
selected = DotDic()
for classifier in args.classifiers:

	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(args.quantiles),
		axis=0)

	selected[classifier] = DotDic()
	for q, quantile in enumerate(args.quantiles):
		selected[classifier][quantile] = DotDic()

		# filtering parameter
		selected[classifier][quantile].cutoff = cutoffs[q]
		cutoff = selected[classifier][quantile].cutoff

		# define test_args for each classifier x quantile
		selected[classifier][quantile].args = DotDic()
		dic = selected[classifier][quantile].args
		dic.update(test_args)
		c = params.background.c[classifier]
		dic.from_, dic.to_ = adaptive_bin(
			X=params.filter(X_=(tX, c), cutoff=cutoff),
			lower=test_args.lower,
			upper=test_args.upper,
			n_bins=test_args.bins)

###################################################
# Plot fits for different polynomial complexities
###################################################
fits(args=args,
	 params=params,
	 ks=[5, test_args.k_star, 35],
	 path='{0}/val'.format(args.target_data_id),
	 filename='fits',
	 alpha=args.alpha)

###################################################
# Plot all CDF curves
###################################################
fig, ax = plt.subplots(nrows=1, ncols=1,
					   figsize=(10, 10),
					   sharex='none',
					   sharey='none')
labels = []
for i, k in enumerate(args.ks):
	labels.append('K={0}'.format(k))
	if k == test_args.k_star:
		labels[i] += ' (selected)'

ax.set_title('P-value CDF per K'.format(test_args.k_star))
plot_cdfs(ax=ax,
		  df=[results[k].pvalues for k in args.ks],
		  labels=labels)
save_fig(cwd=args.cwd,
		 path='{0}/val'.format(args.target_data_id),
		 fig=fig,
		 name='pvalues')

ax.set_xlim([0, args.alpha])
save_fig(cwd=args.cwd,
		 path='{0}/val'.format(args.target_data_id),
		 fig=fig,
		 name='pvalues_restricted')

###################################################
# Plot CI for I(pvalue <= alpha)
###################################################
fig, ax = plt.subplots(nrows=1, ncols=1,
					   figsize=(10, 10),
					   sharex='none',
					   sharey='none')
ax.set_title(
	'Clopper-Pearson CI for I(pvalue<={0}). K={1} selected'.format(
		args.alpha,
		test_args.k_star))
ax.axhline(y=args.alpha,
		   color='black',
		   linestyle='-',
		   label='{0}'.format(args.alpha))
plot_binary_series_with_uncertainty(
	ax,
	x=args.ks,
	values=[results[k].pvalues for k in args.ks],
	color='red',
	alpha=args.alpha)
ax.legend()
save_fig(cwd=args.cwd,
		 path='{0}/val'.format(args.target_data_id),
		 fig=fig,
		 name='level')

###################################################
# Plot measure
###################################################
fig, ax = plt.subplots(nrows=1, ncols=1,
					   figsize=(10, 10),
					   sharex='none',
					   sharey='none')
ax.set_title('Selection measure. K={0} selected'.format(test_args.k_star))
ax.plot(args.ks,
		[results[k].measure for k in args.ks],
		color='black',
		label='Selection measure')
ax.legend()
save_fig(cwd=args.cwd,
		 path='{0}/val'.format(args.target_data_id),
		 fig=fig,
		 name='measure')


# %%
def test_(args, params, classifier, selected, quantiles, lambdas):
	results = DotDic()
	for lambda_ in lambdas:

		results[lambda_] = DotDic()

		for quantile in quantiles:

			# Set cutoff point
			cutoff = selected[classifier][quantile].cutoff
			test_args = selected[classifier][quantile].args
			results[lambda_][quantile] = []

			# Run procedures on all datasets
			print("\nTest Classifier={0} K={1} lambda={2} cutoff={3}\n".format(
				classifier,
				test_args.k,
				lambda_,
				quantile))

			for i in tqdm(range(args.folds), ncols=40):
				X = params.subsample_and_filter(
					n=args.sample_size,
					classifier=classifier,
					lambda_=lambda_,
					cutoff=cutoff)

				test_ = test(args=test_args, X=X)
				pvalue = test_.pvalue
				results[lambda_][quantile].append(pvalue)

			results[lambda_][quantile] = np.array(results[lambda_][quantile])

			clear()

	return results


def power_analysis(params, string):
	path = '{0}/{1}'.format(args.target_data_id, string)

	for classifier in params.classifiers:
		filtering(args=args,
				  params=params,
				  classifier=classifier,
				  lambdas=[0, args.alpha],
				  quantiles=[0.0, 0.5, 0.7],
				  path=path,
				  filename='{0}_filter_restricted'.format(classifier),
				  alpha=args.alpha,
				  k=test_args.k_star)

	if args.use_cache:
		results = load_obj(cwd=args.cwd,
						   path=path,
						   name='power')
	else:
		results = DotDic()
		for classifier in params.classifiers:
			results[classifier] = test_(args=args,
										params=params,
										classifier=classifier,
										selected=selected,
										quantiles=args.quantiles,
										lambdas=lambdas)
		save_obj(cwd=args.cwd,
				 path=path,
				 obj=results,
				 name='power')

	power_per_classifier(
		path=path,
		results=results,
		lambdas=lambdas,
		quantiles=args.quantiles,
		classifiers=args.classifiers,
		labels=['Without Decorrelation',
				'With Decorrelation'])

	clear()

	return results


##################################################
# Validation data analysis
##################################################
results = power_analysis(params=params, string='val')
##################################################
# Test data analysis
##################################################
args.data_id = args.target_data_id
params = load_background(args)
params = load_signal(params)
results = power_analysis(params=params, string='test')

###################################################################
# Plot distribution and cdf of pvalues per quantile
###################################################################

plot = lambda classifier, quantile: plot_cdfs(
	ax=ax,
	df=[results[classifier][lambda_][quantile] for lambda_ in lambdas],
	labels=['$\lambda=${0}'.format(lambda_) for lambda_ in lambdas])

for quantile in args.quantiles:
	fig, axs = plt.subplots(nrows=1, ncols=2,
							figsize=(20, 10),
							sharex='none',
							sharey='none')
	fig.suptitle('CDF of pvalues (BR%={0})'.format(quantile * 100),
				 fontsize=30)

	ax = axs[0]
	ax.set_title('Without Decorrelation', fontsize=30)
	ax.set_title(
		'CDF of pvalues (BR%={0})'.format(quantile * 100))
	plot('class', quantile)

	ax = axs[1]
	ax.set_title('With Decorrelation', fontsize=30)
	ax.set_title(
		'CDF of pvalues (BR%={0})'.format(quantile * 100))
	plot('tclass', quantile)

	save_fig(cwd=args.cwd,
			 path='{0}/test/CDF_per_quantile'.format(args.target_data_id),
			 fig=fig,
			 name='{0}'.format(quantile))

###################################################################
# Total time
###################################################################
print(
	"--- Runtime: %s hours ---" % (round((time.time() - start_time) / 3600, 2)))
