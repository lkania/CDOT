from jax import numpy as np
from tqdm import tqdm
from src import normalize
######################################################################
# Configure matplolib
######################################################################
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

config = {'figure.dpi': 600,
		  'legend.fontsize': 25,
		  'axes.labelsize': 25,
		  'axes.titlesize': 25,
		  'xtick.labelsize': 15,
		  'ytick.labelsize': 15}
pylab.rcParams.update(config)

colors = ['red',
		  'limegreen',
		  'blue',
		  'magenta',
		  'cyan',
		  'darkorange',
		  'grey',
		  'tab:pink',
		  'tab:olive',
		  'purple',
		  'peru']

######################################################################
# Load utilities
######################################################################
from src import bin
from src.stat import binom
from experiments import storage

uniform_bin = lambda X, lower, upper, n_bins: bin.full_uniform_bin(
	n_bins=n_bins)


# See: https://stackoverflow.com/questions/51717199/how-to-adjust-space-between-every-second-row-of-subplots-in-matplotlib
def tight_pairs(n_cols, fig, n_rows):
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

	for ax in fig.axes:
		if hasattr(ax, 'get_subplotspec'):
			ss = ax.get_subplotspec()
			row, col = ss.num1 // n_cols, ss.num1 % n_cols
			if (row % n_rows == 0) and (
					col == 0):  # upper-half row (first subplot)
				y0_upper = ss.get_position(fig).y0
			elif (row % n_rows == 1):  # lower-half row (all subplots)
				x0_low, _, width_low, height_low = ss.get_position(fig).bounds
				ax.set_position(
					pos=[x0_low, y0_upper - height_low, width_low, height_low])

	return fig


def series_with_uncertainty(ax, x, mean,
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
				alpha=0.5,
				fmt=fmt,
				label=label)


def binary_series_with_uncertainty(ax,
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
		lower, mean, upper = binom.clopper_pearson_binomial_ci(
			values=v,
			alpha=alpha)
		means.append(mean)
		lowers.append(lower)
		uppers.append(upper)

	return series_with_uncertainty(ax=ax,
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


def hist_with_uncertainty(ax,
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
	series_with_uncertainty(ax=ax,
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


def hists(ax,
		  info,
		  alpha,
		  tol,
		  lambda_,
		  binning=None,
		  aggregate=None,
		  ax2=None,
		  ax3=None,
		  eps=1e-2):
	methods = info.runs

	ax.set_xlim([0 - eps, 1 + eps])
	ax.set_ylabel('Counts ($\lambda={0}$)'.format(lambda_))
	ax.set_xlabel('Invariant mass')

	if binning is not None:
		# The function assumes that all methods have the same X range
		from_, to_ = binning(
			X=methods[0].X,
			lower=methods[0].test.args.lower,
			upper=methods[0].test.args.upper,
			n_bins=methods[0].test.args.bins)
	else:
		from_ = methods[0].test.args.from_
		to_ = methods[0].test.args.to_

	lower = methods[0].test.args.lower
	upper = methods[0].test.args.upper

	if aggregate is not None:
		# try:
		# Do not aggregate the simulations
		# Instead, just plot the results for the first simulation
		predictions = [info.runs[aggregate].predict_counts(from_=from_, to_=to_)]
		methods = [info.runs[aggregate]]
		# except IndexError as e:
		# 	assert False,"Idx is {}".format(aggregate)
	else:
		predictions = info.predict_counts(from_=from_, to_=to_)

	predictions = np.array(predictions)
	predictions = normalize.threshold_non_neg(predictions, tol=tol)

	count = []
	print('\nBinning observations sequentially\n')
	for i in tqdm(range(len(methods)), ncols=40):
		method = methods[i]
		c = bin.counts(X=method.X, from_=from_, to_=to_)[0]
		assert not np.isnan(c).any()
		count.append(c)
	count = np.array(count, dtype=np.int64)

	assert count.shape == predictions.shape

	pred_lower, pred_mid, pred_upper = binom.bootstrap_percentile_ci(
		values=predictions,
		alpha=alpha)

	count_lower, count_mid, count_upper = binom.bootstrap_percentile_ci(
		values=count,
		alpha=alpha)

	ax.axvline(x=lower, color='green', linestyle='--')
	ax.axvline(x=upper, color='green', linestyle='--', label='Signal region')

	hist_with_uncertainty(
		ax=ax,
		from_=from_,
		to_=to_,
		mean=count_mid,
		lower=count_lower,
		upper=count_upper,
		color='blue',
		label='Data',
		markersize=2)

	label = 'Prediction'  # .format(methods[0].test.args.k)

	hist_with_uncertainty(
		ax=ax,
		from_=from_,
		to_=to_,
		mean=pred_mid,
		lower=pred_lower,
		upper=pred_upper,
		jitter=0,
		color='red',
		markersize=2,
		label=label)

	if ax2 is not None:
		ax.set_xlabel('')
		ax.set_xticks([])
		ax.set_xticks([], minor=True)

		ax2.set_ylabel('Obs / Pred')
		ax2.set_xlabel('Invariant mass')
		ax2.set_xlim([0 - eps, 1 + eps])

		ax2.axvline(x=lower,
					color='green', linestyle='--')
		ax2.axvline(x=upper,
					color='green', linestyle='--')
		ax2.axhline(y=1,
					color='black',
					linestyle='-')
		# ax2.set_ylim(bottom=2)

		# individual confidence intervals for observed / predicted ratio
		ratio_lower, ratio_mid, ratio_upper = binom.bootstrap_percentile_ci(
			values=normalize.safe_ratio(num=count, den=predictions, tol=tol),
			alpha=alpha)

		hist_with_uncertainty(
			ax=ax2,
			from_=from_,
			to_=to_,
			mean=ratio_mid,
			lower=ratio_lower,
			upper=ratio_upper,
			jitter=0,
			color='black',
			markersize=2,
			label='Ratio')

		ax2.set_ylim([0, 1.5])

		# ax2.set_ylim([np.quantile(pred_lower, alpha),
		# 			  np.quantile(pred_upper, 1 - alpha)])

		# ax2.set_ylim([1 - 0.2, 1 + 0.2])
		ax2.legend()

	###################################################################
	# Plot estimates histogram
	###################################################################
	# if ax3 is not None:
	# 	ax3.set_xlabel('P-value')
	# 	ax3.set_ylabel('Counts')
	# 	# ax3.set_yscale('log')
	#
	# 	threshold_ = info.test.threshold
	# 	data_below_threshold = np.mean(np.array(info.stats <= threshold_,
	# 											dtype=np.int32))
	# 	ax3.axvline(x=threshold_,
	# 				color='red',
	# 				linestyle='--',
	# 				label='% data below threshold={0:.2f}'.format(
	# 					round(data_below_threshold, 2)
	# 				))
	#
	# 	ax3.hist(info.stats,
	# 			 alpha=1,
	# 			 bins=int(info.stats.shape[0] / 10),
	# 			 density=False,
	# 			 histtype='step',
	# 			 color='black')
	# 	ax3.legend()

	ax.legend()


def cdfs(ax, df, labels, alpha, eps=1e-2):
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
	ax.set_xlabel('P-value')
	ax.legend()


def save_fig(cwd, path, fig, name):
	base_path = storage.get_path(cwd=cwd, path=path)
	filename = base_path + '{0}.pdf'.format(name)
	fig.savefig(fname=filename, bbox_inches='tight')
	print('\nSaved to {0}'.format(filename))
	plt.close(fig)
