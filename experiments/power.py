######################################################################
# Utilities
######################################################################
from tqdm import tqdm
######################################################################
# Configure matplolib
######################################################################
import distinctipy
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# quick display when debugging
# from src.transform import transform

# trans, tilt_density, _ = transform(a=0, b=None, rate=0.1)
# sns.histplot([params.background.X, trans(params.background.X)])
# sns.histplot(trans(params.background.X))
# plt.show()

plt.rcParams['figure.dpi'] = 600
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
from experiments.builder import load_background, load_signal, filter
from src.test.test import test
from src.stat.binom import clopper_pearson
from src.load import load

######################################################################

# %%
args = parse()
args.folds = 100
args.bins = 100
args.method = 'bin_mle'
args.optimizer = 'dagostini'
args.maxiter = 1000
args.sampling_type = 'subsample'
args.sampling_size = 15000
target_data_id = args.data_id

##################################################
# Experiment parameters
# with this selection of parameters, there is already a difference between
# correlation and non-correlation but there shouldn't be any difference
# maybe the data is already substiatially different for some reason?
##################################################
lambdas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ks = [5, 10, 15, 20, 25, 30, 35, 40]
args.ks = None
args.classifiers = ['class', 'tclass']
##################################################
# Load background data for model selection
##################################################
args.data_id = '{0}/val'.format(target_data_id)
args.lambda_star = 0.0

params = load_background(args)
params = load_signal(args, params)

##################################################
# Set parameters for transforming background
##################################################

args.a = float(np.round(np.min(params.background.X)))
args.b = float(np.round(np.max(params.background.X)))
match target_data_id:
	case '3b' | '4b':
		args.b = None  # i.e. infinity
		args.rate = 0.003
		args.std_signal_region = 1.5
	case 'WTagging':
		args.rate = 3
		args.std_signal_region = 0.5
	case _:
		raise ValueError('Dataset not supported')

##################################################
# Set lower and upper arguments for signal region
# based on the true mean and standard deviation
# of the signal
##################################################
s = load(path='{0}/data/{1}/signal/mass.txt'.format(
	params.cwd,
	target_data_id))

args.mu_star = np.mean(s)
args.sigma_star = np.std(s)
args.lower = args.mu_star - args.std_signal_region * args.sigma_star
args.upper = args.mu_star + args.std_signal_region * args.sigma_star


# Print amount of data outside signal region


##################################################
# Run model selection
##################################################

def select(args, params, quantiles):
	cutoffs = np.quantile(
		params.background.c[args.classifier],
		q=np.array(quantiles),
		axis=0)

	# print("Classifier {0}".format(args.classifier))
	# print("Cutoffs {0}".format(cutoffs))

	selected = DotDic()
	for c, cutoff in enumerate(cutoffs):
		quantile = quantiles[c]
		selected[quantile] = DotDic()
		selected[quantile].cutoff = cutoff

		args.cutoff = cutoff
		Xs = filter(args, params)

		# run test with background complexity K and compute
		# L2 distance between the CDF of the p-values and
		# the uniform CDF
		l2 = []
		pvaluess = []
		inc = np.square((np.arange(params.folds) + 1) / params.folds)
		for k in ks:

			args.k = k
			print("\nValidation Classifier={0} K={1} cutoff={2}\n".format(
				args.classifier, k, quantile))
			pvalues = []
			for i in tqdm(range(params.folds), ncols=40):
				pvalues.append(test(args=args, X=Xs[i]).pvalue)

			# compute l2 distance proxy
			pvalues = np.array(pvalues)
			pvalues = np.sort(pvalues)
			pvalues_diff = np.concatenate(
				(pvalues[1:], np.array([1]))) - pvalues
			l2_ = np.sum(inc * pvalues_diff)
			l2_ += np.mean(np.square(pvalues)) - 2 / 3

			# save results
			l2.append(l2_)
			pvaluess.append(pvalues)

		idx = np.argmin(np.array(l2))
		selected[quantile].k = ks[idx]
		selected[quantile].pvalues = pvaluess[idx]

		clear()

	return selected


selected = DotDic()
for classifier in params.classifiers:
	args.classifier = classifier
	selected[classifier] = select(args, params, quantiles)


# %%

##################################################
# Power analysis
##################################################


def test_(args, params, selected, quantiles, lambdas):
	results = DotDic()
	for lambda_ in lambdas:

		# Set true signal proportion
		args.lambda_star = lambda_
		results[lambda_] = DotDic()

		params = load_signal(args, params)

		for quantile in quantiles:

			# Set cutoff point
			cutoff = selected[args.classifier][quantile].cutoff
			args.k = selected[args.classifier][quantile].k
			args.cutoff = cutoff
			results[lambda_][quantile] = []
			Xs = filter(args, params)

			# Run procedures on all datasets
			print("\nTest Classifier={0} K={1} lambda={2} cutoff={3}\n".format(
				args.classifier,
				args.k,
				args.lambda_star,
				quantile))

			for i in tqdm(range(params.folds), ncols=40):
				pvalue = test(args=args, X=Xs[i]).pvalue
				results[lambda_][quantile].append(pvalue)

			results[lambda_][quantile] = np.array(results[lambda_][quantile])

			clear()

	return results


args.data_id = target_data_id
params = load_background(args)
results = DotDic()
for classifier in params.classifiers:
	args.classifier = classifier
	results[classifier] = test_(args, params, selected, quantiles, lambdas)

# %%
import matplotlib.pylab as pylab

config = {'legend.fontsize': 'xx-large',
		  'figure.figsize': (20, 30),
		  'axes.labelsize': 'xx-large',
		  'axes.titlesize': 'xx-large',
		  'xtick.labelsize': 'x-large',
		  'ytick.labelsize': 'x-large'}
pylab.rcParams.update(config)

colors = distinctipy.get_colors(
	max(len(lambdas), len(quantiles)),
	exclude_colors=[(0, 0, 0), (1, 1, 1),
					(1, 0, 0), (0, 0, 1)],
	rng=0)
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


fig, axs = plt.subplots(nrows=3, ncols=2,
						sharex='none', sharey='none')
row = -1


###################################################################
# Model selection
###################################################################

def plot(ax, selected, args, eps=1e-2):
	selected = selected[args.classifier]

	ax.set_ylim([0 - eps, 1 + eps])
	ax.set_xlim([0 - eps, 1 + eps])
	ax.axline([0, 0], [1, 1], color='red', label='Uniform CDF')

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


row += 1
ax = axs[row, 0]
ax.set_title('Without Decorrelation', fontsize=30)
args.classifier = 'class'
plot(ax=ax, selected=selected, args=args)
ax = axs[row, 1]
ax.set_title('With Decorrelation', fontsize=30)
args.classifier = 'tclass'
plot(ax=ax, selected=selected, args=args)


###################################################################
# Plot power vs threshold
###################################################################

def plot(ax, results, args, eps=1e-2, alpha=0.05):
	ax.set_title('Clopper-Pearson CI for I(Test=1) at alpha=0.05')
	ax.set_xlabel('Background reject percentage (BR%)')
	ax.set_ylabel('Probability of rejecting $\lambda=0$')
	ax.set_ylim([0 - eps, 1 + eps])

	ax.axhline(y=alpha, color='red',
			   linestyle='-', label='{0}'.format(alpha))
	results = results[args.classifier]
	for i, lambda_ in enumerate(lambdas):
		means = []
		lowers = []
		uppers = []
		for quantile in quantiles:
			pvalues = results[lambda_][quantile]
			tests = np.array(pvalues <= 0.05, dtype=np.int32)
			cp = clopper_pearson(n_successes=[np.sum(tests)],
								 n_trials=args.folds,
								 alpha=alpha)[0]
			means.append(np.mean(tests))
			lowers.append(cp[0])
			uppers.append(cp[1])
		plot_series_with_uncertainty(ax, np.array(quantiles) * 100,
									 means, lowers, uppers,
									 color=colors[i],
									 label='$\lambda$={0}'.format(lambda_))

	ax.legend()


row += 1
ax = axs[row, 0]
args.classifier = 'class'
plot(ax=ax, results=results, args=args)
ax = axs[row, 1]
args.classifier = 'tclass'
plot(ax=ax, results=results, args=args)


###################################################################
# Plot distribution and cdf of pvalues
###################################################################


def plot(ax, results, args, quantile, eps=1e-2):
	results = results[args.classifier]
	ax.set_title(
		'CDF of pvalues (BR%={0})'.format(quantile * 100))
	ax.axline([0, 0], [1, 1], color='red', label='Uniform CDF')
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


row += 1
ax = axs[row, 0]
args.classifier = 'class'
plot(ax=ax, results=results, args=args, quantile=0.5)
ax = axs[row, 1]
args.classifier = 'tclass'
plot(ax=ax, results=results, args=args, quantile=0.5)

###################################################################
# Annotate rows
# See https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
###################################################################
# fig.tight_layout()
rows = ['Model\nselection\nwith 3b\nValidation', 'Test', 'Test']
pad = 15  # in pointspad = 5 # in points
for ax, row in zip(axs[:, 0], rows):
	ax.annotate('{}\nData'.format(row), xy=(0, 0.5),
				xytext=(-ax.yaxis.labelpad - pad, 0),
				xycoords=ax.yaxis.label,
				textcoords='offset points',
				size='xx-large', ha='right', va='center')

###################################################################
# Change background color per row
# See https://stackoverflow.com/questions/59751952/python-plot-different-figure-background-color-for-each-row-of-subplots
###################################################################
colors = ['blue', 'white', 'white']
for ax, color in zip(axs[:, 0], colors):
	bbox = ax.get_position()
	rect = matplotlib.patches.Rectangle(
		(0, bbox.y0), 1, bbox.height,
		alpha=0.05,
		color=color,
		zorder=-1)
	fig.add_artist(rect)

###################################################################
# save figure
###################################################################
path = '{0}/summaries/'.format(params.cwd)
Path(params.path).mkdir(parents=True, exist_ok=True)
filename = path + '{0}.pdf'.format(params.data_id)
fig.savefig(fname=filename, bbox_inches='tight')
plt.close(fig)
print('\nSaved to {0}'.format(filename))
