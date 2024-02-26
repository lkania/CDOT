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
from src.test.test import test
from src.stat.binom import clopper_pearson
from src.bin import proportions, adaptive_bin, full_adaptive_bin, \
	uniform_bin as _uniform_bin

uniform_bin = lambda X, lower, upper, n_bins: _uniform_bin(n_bins=n_bins)


######################################################################
# plot utilities
######################################################################

def binary_rv_uncertainty(values, alpha=0.05):
	values_ = np.array(values, dtype=np.int32)
	cp = clopper_pearson(n_successes=[np.sum(values_)],
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
										alpha=0.05,
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

	return plot_series_with_uncertainty(ax, x,
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
			   alpha=0.05,
			   jitter=1e-1,
			   labels=True,
			   markersize=5):
	params = methods[0].params
	from_, to_ = binning(
		X=methods[0].X,
		lower=args.lower,
		upper=args.upper,
		n_bins=args.bins)

	props = proportions(X=methods[0].X, from_=from_, to_=to_)[0]
	counts = props * methods[0].X.shape[0]

	# TODO: replace by poisson means CI
	cis = clopper_pearson(n_successes=counts,
						  n_trials=methods[0].X.shape[0],
						  alpha=alpha)

	if labels:
		ax.set_xlabel('Mass (projected scale)')
		ax.set_ylabel('Normalized counts')

	ax.axvline(x=args.lower,
			   color='red', linestyle='--')
	ax.axvline(x=args.upper,
			   color='red', linestyle='--',
			   label='Signal region')
	plot_hist_with_uncertainty(ax=ax,
							   from_=from_,
							   to_=to_,
							   mean=props,
							   lower=cis[:, 0].reshape(-1),
							   upper=cis[:, 1].reshape(-1),
							   color='black',
							   label='Data (Clopper-Pearson CI)',
							   markersize=2)
	for i, method in enumerate(methods):
		label = 'K={0} p-value={1}'.format(method.k,
										   round(method.pvalue, 2))
		plot_hist_with_uncertainty(
			ax=ax,
			from_=from_,
			to_=to_,
			mean=params.basis.predict(
				gamma=method.gamma_hat,
				k=method.k,
				from_=from_,
				to_=to_).reshape(-1),
			jitter=(i + 1) * jitter * (to_ - from_),
			color=colors[i],
			markersize=markersize,
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


###################################################################
# save results
###################################################################
import pickle


def save_obj(cwd, data_id, obj, name):
	filename = get_path(cwd=cwd, path=data_id) + '{0}.pickle'.format(name)
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('\nSaved to {0}'.format(filename))


def load_obj(cwd, data_id, name):
	filename = get_path(cwd=cwd, path=data_id) + '{0}.pickle'.format(name)
	with open(filename, 'rb') as handle:
		obj = pickle.load(handle)
	return obj


# %%
##################################################
# Experiment parameters
##################################################
args = parse()

args.sample_size = 20000
args.folds = 500

args.bins = 100
args.method = 'bin_mle'
args.optimizer = 'dagostini'
args.fixpoint = 'normal'
args.maxiter = 1000
args.tol = 1e-6

args.signal_region = [0.1, 0.9]  # quantiles for signal region
target_data_id = args.data_id

lambdas = [0, 0.01, 0.02, 0.05]  # add 0.001, 0.005 afterwards
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ks = [5, 10, 15, 20, 25, 30, 35]
args.classifiers = ['class', 'tclass']

assert max(len(lambdas), len(quantiles)) <= len(colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(target_data_id)

params = load_background(args)
params = load_signal(params)

##################################################
# Background transformation parameters
# Our test assumes that data lies between 0 and 1
# hence, you must project the data to that scale
##################################################

match target_data_id:
	case '3b' | '4b':
		from src.background.transform import exponential

		a = float(np.round(np.min(params.background.X)))
		trans = lambda X: exponential.trans(X=X,
											rate=0.003,
											base=a,
											scale=1)

	case 'WTagging':

		trans = lambda X: X

	case _:
		raise ValueError('Dataset not supported')
args.trans = trans

##################################################
# Set lower and upper arguments for signal region
# based on the true mean and standard deviation
# of the signal
##################################################
qs = np.quantile(
	params.signal.X,
	q=np.array(args.signal_region),
	axis=0)
args.lower = qs[0]
args.upper = qs[1]

# Print amount of data outside signal region
prop, _ = proportions(
	params.signal.X,
	np.array([args.lower]),
	np.array([args.upper]))
prop = prop[0]
print(
	'{0:.2f}% of the loaded signal is contained in the signal region'
	.format(prop * 100))

##################################################
# Fix binning for test statistic
##################################################
args.lower = args.trans(args.lower)
args.upper = args.trans(args.upper)
from_, to_ = adaptive_bin(
	X=args.trans(params.background.X),
	lower=args.lower,
	upper=args.upper,
	n_bins=args.bins)
args.from_ = from_
args.to_ = to_


##################################################
# Run model selection for base case, i.e. no cutoff
##################################################

def fits(args,
		 params,
		 ks,
		 path,
		 alpha=0.05):
	fig, axs = plt.subplots(nrows=1,
							ncols=len(ks),
							figsize=(10 * len(ks),
									 10),
							sharex='none',
							sharey='none')

	X = params.subsample(n=args.sample_size, lambda_=0)

	for i, k in enumerate(ks):
		ax = axs[i]
		args.k = k
		test_ = test(args=args, X=args.trans(X))
		plot_hists(ax,
				   uniform_bin,
				   methods=[test_],
				   jitter=0,
				   labels=False,
				   markersize=5,
				   alpha=alpha)

	save_fig(cwd=args.cwd,
			 path=path,
			 fig=fig,
			 name='fits')


fits(args=args,
	 params=params,
	 ks=ks,
	 path='{0}/val'.format(target_data_id))


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


def target_alpha_level(pvalues, alpha=0.05):
	return np.abs(alpha - np.mean(pvalues <= alpha))


def select(args, params, ks, measure, alpha=0.05):
	# run test with background complexity K and compute
	# L2 distance between the CDF of the p-values and
	# the uniform CDF
	results = DotDic()
	for k in ks:
		results[k] = DotDic()

		args.k = k
		print("\nValidation K={}\n".format(k))
		pvalues = []
		for i in tqdm(range(args.folds), ncols=40):
			X = params.subsample(n=args.sample_size, lambda_=0)
			test_ = test(args=args, X=args.trans(X))
			pvalues.append(test_.pvalue)

		# save results
		pvalues = np.array(pvalues)
		results[k].pvalues = pvalues
		results[k].measure = measure(pvalues)

	# Select K that produces the most uniform (in L_2 sense)
	# p-value distribution
	idx = np.argmin(np.array([results[k].measure for k in ks]))
	k_star = ks[idx]

	###################################################
	# Plot all CDF curves
	###################################################
	fig, ax = plt.subplots(nrows=1, ncols=1,
						   figsize=(10, 10),
						   sharex='none',
						   sharey='none')
	labels = []
	for i, k in enumerate(ks):
		labels.append('K={0}'.format(k))
		if k == k_star:
			labels[i] += ' (selected)'

	ax.set_title('P-value CDF per K. K={0} selected'.format(k_star))
	plot_cdfs(ax=ax,
			  df=[results[k].pvalues for k in ks],
			  labels=labels)
	save_fig(cwd=args.cwd,
			 path='{0}/val'.format(target_data_id),
			 fig=fig,
			 name='pvalues')

	###################################################
	# Plot number of calibration at alpha level
	###################################################
	fig, ax = plt.subplots(nrows=1, ncols=1,
						   figsize=(10, 10),
						   sharex='none',
						   sharey='none')
	ax.set_title('Clopper-Pearson CI for I(pvalue<={0}). K={1} selected'.format(
		alpha,
		k_star))
	ax.axhline(y=alpha,
			   color='black',
			   linestyle='-',
			   label='{0}'.format(alpha))
	plot_binary_series_with_uncertainty(ax,
										x=ks,
										values=[results[k].pvalues for k in ks],
										color='red')
	ax.legend()
	save_fig(cwd=args.cwd,
			 path='{0}/val'.format(target_data_id),
			 fig=fig,
			 name='level')

	###################################################
	# Plot measure
	###################################################
	fig, ax = plt.subplots(nrows=1, ncols=1,
						   figsize=(10, 10),
						   sharex='none',
						   sharey='none')
	ax.set_title('Selection measure. K={0} selected'.format(alpha, k_star))
	ax.plot(ks,
			[results[k].measure for k in ks],
			color='black')
	ax.legend()
	save_fig(cwd=args.cwd,
			 path='{0}/val'.format(target_data_id),
			 fig=fig,
			 name='measure')

	clear()

	return k_star


##################################################
# We find the right complexity for our test statistic
# without any sort of signal enriched procedure
##################################################

# TODO: improve selection for alpha=0.05
k_star = select(args=args,
				params=params,
				ks=ks,
				measure=target_alpha_level)
print("\nSelected K={0}\n".format(k_star))

selected = DotDic()
for classifier in args.classifiers:

	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(quantiles),
		axis=0)

	selected[classifier] = DotDic()
	for q, quantile in enumerate(quantiles):
		selected[classifier][quantile] = DotDic()
		selected[classifier][quantile].cutoff = cutoffs[q]
		selected[classifier][quantile].k = k_star


def filtering(args,
			  params,
			  classifier,
			  lambdas,
			  quantiles,
			  path,
			  alpha=0.05,
			  k=None):
	assert k is not None

	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(quantiles),
		axis=0)

	##################################################
	# Plot datasets with classifier filter
	##################################################
	fig, axs = plt.subplots(nrows=len(lambdas),
							ncols=len(quantiles),
							figsize=(10 * len(quantiles),
									 10 * len(lambdas)),
							sharex='none',
							sharey='none')
	fig.suptitle('Uniform binning (One instance) for {0}'.format(classifier),
				 fontsize=30)

	for l, lambda_ in enumerate(lambdas):
		for c, cutoff in enumerate(cutoffs):

			quantile = quantiles[c]

			X = params.subsample_and_filter(
				n=args.sample_size,
				classifier=classifier,
				lambda_=lambda_,
				cutoff=cutoff)

			ax = axs[l, c]
			if l == 0:
				ax.set_title('Quantile={0}'.format(quantile))
			if c == 0:
				ax.set_ylabel('$\lambda={0}$ Normalized counts'.format(lambda_))
			if c == len(quantiles) - 1:
				ax.set_xlabel('Mass (projected scale)')

			args.k = k
			test_ = test(args=args, X=args.trans(X))
			plot_hists(ax,
					   uniform_bin,
					   methods=[test_],
					   jitter=0,
					   labels=False,
					   markersize=5,
					   alpha=alpha)

	save_fig(cwd=args.cwd, path=path,
			 fig=fig,
			 name='{0}_filter'.format(classifier))


def test_(args, params, classifier, selected, quantiles, lambdas):
	results = DotDic()
	for lambda_ in lambdas:

		results[lambda_] = DotDic()

		for quantile in quantiles:

			# Set cutoff point
			cutoff = selected[classifier][quantile].cutoff
			args.k = selected[classifier][quantile].k
			results[lambda_][quantile] = []

			# Run procedures on all datasets
			print("\nTest Classifier={0} K={1} lambda={2} cutoff={3}\n".format(
				classifier,
				args.k,
				lambda_,
				quantile))

			for i in tqdm(range(args.folds), ncols=40):
				X = params.subsample_and_filter(
					n=args.sample_size,
					classifier=classifier,
					lambda_=lambda_,
					cutoff=cutoff)

				test_ = test(args=args, X=args.trans(X))
				pvalue = test_.pvalue
				results[lambda_][quantile].append(pvalue)

			results[lambda_][quantile] = np.array(results[lambda_][quantile])

			clear()

	return results


def power(ax, results, lambdas, quantiles, eps=1e-2, alpha=0.05):
	ax.set_title('Clopper-Pearson CI for I(Test=1) at alpha=0.05')
	ax.set_xlabel('Background reject percentage (BR%)')
	ax.set_ylabel('Probability of rejecting $\lambda=0$')
	ax.set_ylim([0 - eps, 1 + eps])

	ax.axhline(y=alpha, color='black',
			   linestyle='-', label='{0}'.format(alpha))
	for i, lambda_ in enumerate(lambdas):
		plot_binary_series_with_uncertainty(
			ax,
			x=np.array(quantiles) * 100,
			values=[
				np.array(results[lambda_][quantile] <= alpha, dtype=np.int32)
				for lambda_ in lambdas],
			color=colors[i],
			label='$\lambda$={0}'.format(lambda_))

	ax.legend()


def power_per_classifier(path, results, lambdas, quantiles):
	fig, axs = plt.subplots(nrows=1, ncols=2,
							figsize=(20, 10),
							sharex='none',
							sharey='none')
	ax = axs[0]
	ax.set_title('Without Decorrelation', fontsize=30)
	classifier = 'class'
	power(ax=ax,
		  results=results[classifier],
		  lambdas=lambdas,
		  quantiles=quantiles)
	ax = axs[1]
	ax.set_title('With Decorrelation', fontsize=30)
	classifier = 'tclass'
	power(ax=ax,
		  results=results[classifier],
		  lambdas=lambdas,
		  quantiles=quantiles)
	save_fig(cwd=args.cwd, path=path, fig=fig, name='power')


##################################################
# Validation data analysis
##################################################

for classifier in params.classifiers:
	filtering(args=args,
			  params=params,
			  classifier=classifier,
			  lambdas=lambdas,
			  quantiles=quantiles,
			  path='{0}/val'.format(target_data_id),
			  alpha=0.05,
			  k=k_star)

results = DotDic()
for classifier in params.classifiers:
	results[classifier] = test_(args=args,
								params=params,
								classifier=classifier,
								selected=selected,
								quantiles=quantiles,
								lambdas=lambdas)

power_per_classifier(path='{0}/val'.format(target_data_id),
					 results=results,
					 lambdas=lambdas,
					 quantiles=quantiles)

##################################################
# Test data analysis
##################################################

args.data_id = target_data_id
params = load_background(args)
params = load_signal(params)

for classifier in params.classifiers:
	filtering(args=args,
			  params=params,
			  classifier=classifier,
			  lambdas=lambdas,
			  quantiles=quantiles,
			  path='{0}/test'.format(target_data_id),
			  alpha=0.05,
			  k=k_star)

results = DotDic()
for classifier in params.classifiers:
	results[classifier] = test_(args=args,
								params=params,
								classifier=classifier,
								selected=selected,
								quantiles=quantiles,
								lambdas=lambdas)

power_per_classifier(path='{0}/test'.format(target_data_id),
					 results=results,
					 lambdas=lambdas,
					 quantiles=quantiles)

###################################################################
# Plot distribution and cdf of pvalues per quantile
###################################################################

plot = lambda classifier, quantile: plot_cdfs(
	ax=ax,
	df=[results[classifier][lambda_][quantile] for lambda_ in lambdas],
	labels=['$\lambda=${0}'.format(lambda_) for lambda_ in lambdas])

for quantile in quantiles:
	fig, axs = plt.subplots(nrows=1, ncols=2,
							figsize=(20, 10),
							sharex='none',
							sharey='none')
	fig.suptitle('CDF of pvalues (BR%={0})'.format(quantile * 100), fontsize=30)

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
			 path='{0}/test/CDF_per_quantile'.format(target_data_id),
			 fig=fig,
			 name='{0}'.format(quantile))

###################################################################
# Total time
###################################################################
print(
	"--- Runtime: %s hours ---" % (round((time.time() - start_time) / 3600, 2)))
