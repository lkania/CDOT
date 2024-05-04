from jax import numpy as np
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
# Load utilities
######################################################################
from src import bin
from src.stat.binom import exact_binomial_ci, exact_poisson_ci
from experiments import storage
from src.normalize import threshold_non_neg


def binomial_ci(values, alpha):
	values_ = np.array(values, dtype=np.int32)
	cp = exact_binomial_ci(n_successes=[np.sum(values_)],
						   n_trials=values_.shape[0],
						   alpha=alpha)[0]
	mean = np.mean(values_)
	lower = cp[0]
	upper = cp[1]

	return lower, mean, upper


# TODO: replace mean by refit to all data
def boostrap_pivotal_ci(values, alpha):
	values = np.array(values)
	mean = np.mean(values, axis=0)
	lower = 2 * mean - np.quantile(values, q=1 - alpha / 2, axis=0)
	upper = 2 * mean - np.quantile(values, q=alpha / 2, axis=0)

	midpoint = (lower + upper) / 2

	return lower, midpoint, upper


def bootstrap_percentile_ci(values, alpha, tol=1e-12):
	values = np.array(values)

	lower = np.quantile(values, q=alpha / 2, axis=0)
	lower = normalize.threshold(lower, tol=tol)

	upper = np.quantile(values, q=1 - alpha / 2, axis=0)
	upper = normalize.threshold(upper, tol=tol)

	midpoint = np.quantile(values, q=0.5, axis=0)
	midpoint = normalize.threshold(midpoint, tol=tol)

	return lower, midpoint, upper


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
		lower, mean, upper = binomial_ci(values=v, alpha=alpha)
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


# TODO: produce plots restricting the ax2 axis and not
# restricting it (ax2 and ax3) global and local picture
def hists(ax,
		  methods,
		  alpha,
		  binning=None,
		  ax2=None,
		  eps=1e-2):
	if binning is not None:
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

	predictions = []
	pvalues = []
	count = []

	for i, method in enumerate(methods):
		sample_size = methods[0].X.shape[0]
		p = method.predict(from_=from_, to_=to_).reshape(-1) * sample_size
		assert not np.isnan(p).any()
		predictions.append(p)

		pvalues.append(method.pvalue)

		c = bin.counts(X=method.X, from_=from_, to_=to_)[0]
		assert not np.isnan(c).any()
		count.append(c)

	predictions = np.array(predictions)
	count = np.array(count, dtype=np.int32)

	if len(methods) > 1:

		pred_lower, pred_mid, pred_upper = bootstrap_percentile_ci(
			values=predictions,
			alpha=alpha)
		pvalue_lower, _, pvalue_upper = bootstrap_percentile_ci(
			values=pvalues,
			alpha=alpha)

	else:
		pred_mid = predictions.reshape(-1)
		pred_lower = None
		pred_upper = None
		pvalue_lower = pvalues[0]
		pvalue_upper = pvalues[0]

	count_mean = np.mean(count, axis=0)
	count_pred_ratio = count / predictions
	cis = exact_poisson_ci(n_events=count, alpha=alpha)
	count_lower = cis[:, 0].reshape(-1)
	count_upper = cis[:, 1].reshape(-1)

	ax.axvline(x=lower, color='red', linestyle='--')
	ax.axvline(x=upper, color='red', linestyle='--', label='Signal region')

	hist_with_uncertainty(
		ax=ax,
		from_=from_,
		to_=to_,
		mean=count_mean,
		lower=count_lower,
		upper=count_upper,
		color='black',
		label='Data (Exact Poisson CI)',
		markersize=2)

	label = 'K={0} p-value=[{1:g},{2:g}]'.format(
		methods[0].test.args.k,
		round(pvalue_lower, 2),
		round(pvalue_upper, 2))

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
		ax2.axvline(x=lower,
					color='red', linestyle='--')
		ax2.axvline(x=upper,
					color='red', linestyle='--',
					label='Signal region')
		ax2.axhline(y=1,
					color='black',
					linestyle='-')
		# ax2.set_ylim(bottom=0)

		if pred_lower is not None and pred_upper is not None:
			pred_lower, pred_mid, pred_upper = bootstrap_percentile_ci(
				values=count_pred_ratio,
				alpha=alpha)

			pred_upper = threshold_non_neg(pred_upper, tol=0.0)
			pred_mid = threshold_non_neg(pred_mid, tol=0.0)
			pred_lower = threshold_non_neg(pred_lower, tol=0.0)
		else:
			pred_mid = count_pred_ratio.reshape(-1)

		hist_with_uncertainty(
			ax=ax2,
			from_=from_,
			to_=to_,
			mean=pred_mid,
			lower=pred_lower,
			upper=pred_upper,
			jitter=0,
			color='red',
			markersize=2,
			label=label)

	# We don't display large outliers
	# We check that our proposed new upper limit
	# is smaller than the current upper limit in
	# order to not overwrite other upper limits
	# when sharing y axes with other plots
	# l, u = ax2.get_ylim()
	# u_proposal = quantile_interval(values=pred_upper, alpha=alpha)[2]
	# ax2.set_ylim(bottom=np.maximum(l, 0 - eps),
	# 			 top=np.maximum(np.maximum(u, u_proposal), 1 + eps))

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
			label='{0} {1}-qtile={2}'.format(labels[i],
											 alpha,
											 round(np.quantile(d, q=alpha), 2)
											 )
		)

	ax.set_ylabel('Cumulative probability')
	ax.set_xlabel('pvalue')
	ax.legend()


def save_fig(cwd, path, fig, name):
	filename = storage.get_path(cwd=cwd, path=path) + '{0}.pdf'.format(name)
	fig.savefig(fname=filename, bbox_inches='tight')
	plt.close(fig)
	print('\nSaved to {0}'.format(filename))
