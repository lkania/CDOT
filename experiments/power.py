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


def plot_hist_with_uncertainty(ax,
							   from_,
							   to_,
							   mean,
							   lower=None,
							   upper=None,
							   jitter=0,
							   color='black',
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
								 markersize=2,
								 elinewidth=1,
								 capsize=1,
								 fmt='o')


def plot_hists(ax, binning, ks, methods, alpha=0.05, jitter=1e-1):
	params = methods[0].params
	from_, to_ = binning(
		X=methods[0].tX,
		lower=params.tlower,
		upper=params.tupper,
		n_bins=params.bins)

	props = proportions(X=methods[0].tX, from_=from_, to_=to_)[0]
	counts = props * methods[0].tX.shape[0]

	cis = clopper_pearson(n_successes=counts,
						  n_trials=methods[0].tX.shape[0],
						  alpha=alpha)

	ax.set_xlabel('Mass (projected scale)')
	ax.set_ylabel('Normalized counts')
	ax.axvline(x=params.tlower,
			   color='red', linestyle='--')
	ax.axvline(x=params.tupper,
			   color='red', linestyle='--',
			   label='Signal region')
	plot_hist_with_uncertainty(ax=ax,
							   from_=from_,
							   to_=to_,
							   mean=props,
							   lower=cis[:, 0].reshape(-1),
							   upper=cis[:, 1].reshape(-1),
							   color='black',
							   label='Data (Clopper-Pearson CI)')
	for i, k in enumerate(ks):
		m = methods[i]
		label = 'Model selection' if k is None else 'K={0}'.format(k)
		plot_hist_with_uncertainty(
			ax=ax,
			from_=from_,
			to_=to_,
			mean=params.basis.predict(
				gamma=m.gamma_hat,
				k=m.k,
				from_=from_,
				to_=to_).reshape(-1),
			jitter=(i + 1) * jitter * (to_ - from_),
			color=colors[i],
			label=label)

		if k is None:
			ax.axvline(x=m.model_selection.from_,
					   color='blue', linestyle='--')
			ax.axvline(x=m.model_selection.to_,
					   color='blue', linestyle='--',
					   label='Model selection region')

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
from src.load import load
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
args.folds = 3  # 500  # 500 gives nice results
args.bins = 100
args.method = 'bin_mle'
args.optimizer = 'dagostini'
args.maxiter = 1000
args.sample_size = 15000
args.signal_region = [0.2, 0.8]  # quantiles for signal region
target_data_id = args.data_id

lambdas = [0.0, 0.01, 0.02, 0.05]  # add 0.001, 0.005 afterwards
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # [0.0, 0.1, 0.4,
# 0.7]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ks = [5, 10, 15, 20, 25]  # , 30, 35, 40]
args.ks = None
args.classifiers = ['class', 'tclass']

assert max(len(lambdas), len(quantiles)) <= len(colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(target_data_id)

params = load_background(args)

##################################################
# Background transformation parameters
# Our test assumes that data lies between 0 and 1
# hence, you must project the data to that scale
##################################################

match target_data_id:
	case '3b' | '4b':
		from src.background.transform import exponential

		a = float(np.round(np.min(params.background.X)))
		b = None  # i.e. infinity
		rate = 0.003
		trans, tilt_density, _ = exponential.transform(
			a=a, b=b, rate=rate)

	case 'WTagging':

		# args.rate = 3, a = min b = max
		# we use the identity mapping
		trans = lambda X: X
		tilt_density = lambda density, X: density(X=X)

	case _:
		raise ValueError('Dataset not supported')
args.trans = trans
args.tilt_density = tilt_density

##################################################
# Set lower and upper arguments for signal region
# based on the true mean and standard deviation
# of the signal
##################################################
s = load(path='{0}/data/{1}/val/signal/mass.txt'.format(
	params.cwd,
	target_data_id))
qs = np.quantile(s, q=np.array(args.signal_region), axis=0)
args.lower = qs[0]
args.upper = qs[1]

# Print amount of data outside signal region
prop, _ = proportions(s, np.array([args.lower]), np.array([args.upper]))
prop = prop[0]
print(
	'{0:.2f}% of the loaded signal is contained in the signal region'
	.format(prop * 100))


##################################################
# Run model selection
##################################################

def select(args, params, classifier, quantiles, alpha=0.05):
	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(quantiles),
		axis=0)

	##################################################
	# Plot datasets with classifier filter
	##################################################
	fig, axs = plt.subplots(nrows=1,
							ncols=len(quantiles),
							figsize=(10 * len(quantiles), 10),
							sharex='none',
							sharey='none')
	fig.suptitle('Uniform binning (One instance) for {0}'.format(classifier),
				 fontsize=30)
	for c, cutoff in enumerate(cutoffs):
		quantile = quantiles[c]
		X = params.subsample_and_filter(
			n=args.sample_size,
			classifier=classifier,
			lambda_=0,
			cutoff=cutoff)
		ax = axs[c]
		ax.set_title('Quantile={0}'.format(quantile))

		from_, to_ = _uniform_bin(
			from_=np.min(X),
			to_=np.max(X),
			n_bins=args.bins)

		props = proportions(X=X, from_=from_, to_=to_)[0]
		counts = props * X.shape[0]
		cis = clopper_pearson(n_successes=counts,
							  n_trials=X.shape[0],
							  alpha=alpha)

		ax.set_xlabel('Mass (projected scale)')
		ax.set_ylabel('Normalized counts')
		ax.axvline(x=args.lower,
				   color='red',
				   linestyle='--')
		ax.axvline(x=args.upper,
				   color='red',
				   linestyle='--',
				   label='Signal region')
		plot_hist_with_uncertainty(ax=ax,
								   from_=from_,
								   to_=to_,
								   mean=props,
								   lower=cis[:, 0].reshape(-1),
								   upper=cis[:, 1].reshape(-1),
								   color='black',
								   label='Data (Clopper-Pearson CI)')
		ax.legend()
	save_fig(cwd=args.cwd,
			 path='{0}/val/{1}'.format(
				 target_data_id,
				 classifier),
			 fig=fig,
			 name='filtering')

	selected = DotDic()
	for c, cutoff in enumerate(cutoffs):
		quantile = quantiles[c]
		selected[quantile] = DotDic()
		selected[quantile].cutoff = cutoff

		# run test with background complexity K and compute
		# L2 distance between the CDF of the p-values and
		# the uniform CDF
		l2 = []
		pvaluess = []
		inc = (np.arange(args.folds) + 1) / args.folds
		for k in ks:
			args.k = k
			print("\nValidation Classifier={0} K={1} cutoff={2}\n".format(
				classifier, k, quantile))
			pvalues = []
			for i in tqdm(range(args.folds), ncols=40):
				X = params.subsample_and_filter(
					n=args.sample_size,
					classifier=classifier,
					lambda_=0,
					cutoff=cutoff)

				test_ = test(args=args, X=X)
				pvalues.append(test_.pvalue)

				###################################################
				# Plot first fit for every classifier/quantile/K
				###################################################
				if i == 0:
					fig, axs = plt.subplots(nrows=1,
											ncols=3,
											figsize=(30, 10),
											sharex='none',
											sharey='none')
					ax = axs[0]
					ax.set_title(
						'Adaptive binning (Control Region) (One instance)')
					plot_hists(ax, adaptive_bin, ks=[k], methods=[test_],
							   jitter=0)

					ax = axs[1]
					ax.set_title('Adaptive binning (One instance)')
					plot_hists(ax, full_adaptive_bin, ks=[k], methods=[test_],
							   jitter=0)

					ax = axs[2]
					ax.set_title('Uniform binning (One instance)')
					plot_hists(ax, uniform_bin, ks=[k], methods=[test_],
							   jitter=0)

					save_fig(cwd=args.cwd,
							 path='{0}/val/{1}/{2}'.format(
								 target_data_id,
								 classifier,
								 quantile),
							 fig=fig,
							 name='{0}'.format(k))

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
			pvalues = np.array(pvalues)
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

			# save results
			l2.append(A + C)
			pvaluess.append(pvalues)

		# Select K that produces the most uniform (in L_2 sense)
		# p-value distribution
		idx = np.argmin(np.array(l2))
		selected[quantile].k = ks[idx]
		selected[quantile].pvalues = pvaluess[idx]

		###################################################
		# Plot all CDF curves for given quantile
		###################################################
		fig, ax = plt.subplots(nrows=1, ncols=1,
							   figsize=(10, 10),
							   sharex='none',
							   sharey='none')
		plot_cdfs(ax=ax, df=pvaluess, labels=['K={0}'.format(k) for k in ks])
		save_fig(cwd=args.cwd,
				 path='{0}/val/{1}/{2}'.format(
					 target_data_id,
					 classifier,
					 quantile),
				 fig=fig,
				 name='pvalues')

		clear()

	return selected


selected = DotDic()
for classifier in params.classifiers:
	selected[classifier] = select(args=args,
								  params=params,
								  classifier=classifier,
								  quantiles=quantiles)

# save_obj(cwd=args.cwd, data_id=target_data_id, obj=selected, name='selected')

##################################################
# Plot model selection on validation data
##################################################

print(
	"--- Runtime until model selection: %s hours ---" % (
		round((time.time() - start_time) / 3600, 2)))

fig, axs = plt.subplots(nrows=1, ncols=2,
						figsize=(20, 10),
						sharex='none',
						sharey='none')

plot = lambda classifier: plot_cdfs(
	ax=ax,
	df=[selected[classifier][quantile].pvalues
		for quantile in quantiles],
	labels=[
		'BR%={0} K={1}'.format(quantile * 100,
							   selected[classifier][quantile].k)
		for quantile in quantiles])

ax = axs[0]
ax.set_title('Without Decorrelation', fontsize=30)
plot('class')

ax = axs[1]
ax.set_title('With Decorrelation', fontsize=30)
plot('tclass')

save_fig(cwd=args.cwd, path=target_data_id, fig=fig, name='selection')


##################################################
# Power analysis
##################################################


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

				test_ = test(args=args, X=X)
				pvalue = test_.pvalue
				results[lambda_][quantile].append(pvalue)

				###################################################
				# Plot first fit for every classifier/quantile/lambda
				###################################################
				if i == 0:
					fig, axs = plt.subplots(nrows=1,
											ncols=3,
											figsize=(30, 10),
											sharex='none',
											sharey='none')
					ax = axs[0]
					ax.set_title(
						'Adaptive binning (Control Region) (One instance)')
					plot_hists(ax, adaptive_bin, ks=[args.k], methods=[test_],
							   jitter=0)

					ax = axs[1]
					ax.set_title('Adaptive binning (One instance)')
					plot_hists(ax, full_adaptive_bin, ks=[args.k],
							   methods=[test_],
							   jitter=0)

					ax = axs[2]
					ax.set_title('Uniform binning (One instance)')
					plot_hists(ax, uniform_bin, ks=[args.k],
							   methods=[test_],
							   jitter=0)

					save_fig(cwd=args.cwd,
							 path='{0}/test/{1}/{2}'.format(
								 target_data_id,
								 classifier,
								 quantile),
							 fig=fig,
							 name='{0}'.format(lambda_))

			results[lambda_][quantile] = np.array(results[lambda_][quantile])

			clear()

	return results


args.data_id = target_data_id
params = load_background(args)
params = load_signal(params)
results = DotDic()
for classifier in params.classifiers:
	results[classifier] = test_(args=args,
								params=params,
								classifier=classifier,
								selected=selected,
								quantiles=quantiles,
								lambdas=lambdas)


###################################################################
# Plot power vs threshold
###################################################################

def plot(ax, results, classifier, eps=1e-2, alpha=0.05):
	ax.set_title('Clopper-Pearson CI for I(Test=1) at alpha=0.05')
	ax.set_xlabel('Background reject percentage (BR%)')
	ax.set_ylabel('Probability of rejecting $\lambda=0$')
	ax.set_ylim([0 - eps, 1 + eps])

	ax.axhline(y=alpha, color='black',
			   linestyle='-', label='{0}'.format(alpha))
	results = results[classifier]
	for i, lambda_ in enumerate(lambdas):
		means = []
		lowers = []
		uppers = []
		for quantile in quantiles:
			pvalues = results[lambda_][quantile]
			tests = np.array(pvalues <= 0.05, dtype=np.int32)
			cp = clopper_pearson(n_successes=[np.sum(tests)],
								 n_trials=tests.shape[0],
								 alpha=alpha)[0]
			means.append(np.mean(tests))
			lowers.append(cp[0])
			uppers.append(cp[1])
		plot_series_with_uncertainty(ax, np.array(quantiles) * 100,
									 means, lowers, uppers,
									 color=colors[i],
									 label='$\lambda$={0}'.format(lambda_))

	ax.legend()


fig, axs = plt.subplots(nrows=1, ncols=2,
						figsize=(20, 10),
						sharex='none',
						sharey='none')
ax = axs[0]
ax.set_title('Without Decorrelation', fontsize=30)
plot(ax=ax, results=results, classifier='class')
ax = axs[1]
ax.set_title('With Decorrelation', fontsize=30)
plot(ax=ax, results=results, classifier='tclass')
save_fig(cwd=args.cwd,
		 path='{0}/test'.format(target_data_id),
		 fig=fig, name='power')

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
