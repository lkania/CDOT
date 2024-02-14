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
import pickle
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
from src.bin import proportions


######################################################################


###################################################################
# save figure
###################################################################
def get_path(params):
	path = '{0}/results/{1}/'.format(params.cwd, params.data_id)
	Path(params.path).mkdir(parents=True, exist_ok=True)
	return path


def save_fig(params, fig, name):
	filename = get_path(params) + '{0}.pdf'.format(name)
	fig.savefig(fname=filename, bbox_inches='tight')
	plt.close(fig)
	print('\nSaved to {0}'.format(filename))


def save_obj(params, obj, name):
	filename = get_path(params) + '{0}.pickle'.format(name)
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(params, name):
	filename = get_path(params) + '{0}.pickle'.format(name)
	with open(filename, 'rb') as handle:
		obj = pickle.load(handle)
	return obj


# %%
##################################################
# Experiment parameters
##################################################
args = parse()
args.folds = 100  # 500 gives nice results
args.bins = 100
args.method = 'bin_mle'
args.optimizer = 'dagostini'
args.maxiter = 1000
args.sample_size = 15000
args.signal_region = [0.2, 0.8]  # quantiles for signal region
target_data_id = args.data_id

lambdas = [0.0, 0.01, 0.02, 0.05]  # add 0.001, 0.005 afterwards
quantiles = [0.0, 0.1, 0.4,
			 0.7]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ks = [5, 10, 15, 20, 25, 30, 35, 40]
args.ks = None
args.classifiers = ['class', 'tclass']

assert max(len(lambdas), len(quantiles)) <= len(colors)
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(target_data_id)

params = load_background(args)

##################################################
# Set parameters for transforming background
##################################################

args.a = float(np.round(np.min(params.background.X)))
args.b = float(np.round(np.max(params.background.X)))
match target_data_id:
	case '3b' | '4b':
		args.b = None  # i.e. infinity
		args.rate = 0.003
	case 'WTagging':
		args.rate = 3
	case _:
		raise ValueError('Dataset not supported')

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

def select(args, params, classifier, quantiles):
	cutoffs = np.quantile(
		params.background.c[classifier],
		q=np.array(quantiles),
		axis=0)

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

				pvalues.append(test(args=args, X=X).pvalue)

			# compute l2 distance proxy
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

		idx = np.argmin(np.array(l2))
		selected[quantile].k = ks[idx]
		selected[quantile].pvalues = pvaluess[idx]

		clear()

	return selected


selected = DotDic()
for classifier in params.classifiers:
	selected[classifier] = select(args=args,
								  params=params,
								  classifier=classifier,
								  quantiles=quantiles)

##################################################
# Plot model selection on validation data
##################################################

print(
	"--- Runtime until model selection: %s hours ---" % (
		round((time.time() - start_time) / 3600, 2)))


def plot(ax, selected, classifier, eps=1e-2):
	selected = selected[classifier]

	ax.set_ylim([0 - eps, 1 + eps])
	ax.set_xlim([0 - eps, 1 + eps])
	ax.axline([0, 0], [1, 1], color='black', label='Uniform CDF')

	for i, quantile in enumerate(quantiles):
		pvalues = selected[quantile].pvalues
		sns.ecdfplot(
			data=pvalues,
			hue_norm=(0, 1),
			legend=False,
			ax=ax,
			color=colors[i],
			alpha=1,
			label='BR%={0} K={1}'.format(quantile * 100,
										 selected[quantile].k))

	ax.set_ylabel('Cumulative probability')
	ax.set_xlabel('pvalue')
	ax.legend()


params.data_id = target_data_id
fig, axs = plt.subplots(nrows=1, ncols=2,
						figsize=(20, 10),
						sharex='none',
						sharey='none')
ax = axs[0]
ax.set_title('Without Decorrelation', fontsize=30)
plot(ax=ax, selected=selected, classifier='class')
ax = axs[1]
ax.set_title('With Decorrelation', fontsize=30)
plot(ax=ax, selected=selected, classifier='tclass')
save_fig(params, fig, 'selection')


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

				pvalue = test(args=args, X=X).pvalue
				results[lambda_][quantile].append(pvalue)

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

# %%

row = -1


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
save_fig(params, fig, 'power')


###################################################################
# Plot distribution and cdf of pvalues
###################################################################


def plot(ax, results, classifier, quantile, eps=1e-2):
	results = results[classifier]
	ax.set_title(
		'CDF of pvalues (BR%={0})'.format(quantile * 100))
	ax.axline([0, 0], [1, 1], color='black', label='Uniform CDF')
	ax.set_ylim([0 - eps, 1 + eps])
	ax.set_xlim([0 - eps, 1 + eps])

	for i, lambda_ in enumerate(lambdas):
		pvalues = results[lambda_][quantile]
		sns.ecdfplot(
			data=pvalues,
			hue_norm=(0, 1),
			legend=False,
			ax=ax,
			color=colors[i],
			alpha=1,
			label='$\lambda=${0}'.format(lambda_))

	ax.set_ylabel('Cumulative probability')
	ax.set_xlabel('pvalue')
	ax.legend()


fig, axs = plt.subplots(nrows=1, ncols=2,
						figsize=(20, 10),
						sharex='none',
						sharey='none')
ax = axs[0]
ax.set_title('Without Decorrelation', fontsize=30)
plot(ax=ax, results=results, classifier='class', quantile=0.4)
ax = axs[1]
ax.set_title('With Decorrelation', fontsize=30)
plot(ax=ax, results=results, classifier='tclass', quantile=0.4)
save_fig(params, fig, 'power-CDF')

###################################################################
# Annotate rows
# See https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
###################################################################
# fig.tight_layout()
# rows = ['Model\nselection\nwith 3b\nValidation', 'Test', 'Test']
# pad = 15  # in pointspad = 5 # in points
# for ax, row in zip(axs[:, 0], rows):
# 	ax.annotate('{}\nData'.format(row), xy=(0, 0.5),
# 				xytext=(-ax.yaxis.labelpad - pad, 0),
# 				xycoords=ax.yaxis.label,
# 				textcoords='offset points',
# 				size='xx-large', ha='right', va='center')

###################################################################
# Change background color per row
# See https://stackoverflow.com/questions/59751952/python-plot-different-figure-background-color-for-each-row-of-subplots
###################################################################
# colors = ['blue', 'white', 'white']
# for ax, color in zip(axs[:, 0], colors):
# 	bbox = ax.get_position()
# 	rect = matplotlib.patches.Rectangle(
# 		(0, bbox.y0), 1, bbox.height,
# 		alpha=0.05,
# 		color=color,
# 		zorder=-1)
# 	fig.add_artist(rect)

###################################################################
# Total time
###################################################################
print(
	"--- Runtime: %s hours ---" % (round((time.time() - start_time) / 3600, 2)))
