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

linetypes = [
	'-',  # solid
	(0, (6, 3)),  # medium dash
	(0, (2, 2)),  # short dash
	(0, (6, 2, 2, 2)),  # dash–dot
	(0, (10, 3)),  # long dash
	(0, (3, 3, 1.5, 3)),  # short dash–dot
	(0, (10, 2, 2, 2, 2, 2)),  # long dash–dot–dot
	(0, (1.5, 2.5)),  # dotted (print-safe spacing)
	(0, (8, 4, 1.5, 4)),  # wide dash–dot
	(0, (3, 2, 3, 4)),  # asymmetric dash pattern
	(0, (12, 3, 3, 3)),  # very long + short dash
]

markers = [
	'o',  # circle
	's',  # square
	'^',  # triangle up
	'D',  # diamond
	'v',  # triangle down
	'P',  # filled plus
	'X',  # filled x
	'<',  # triangle left
	'>',  # triangle right
	'h',  # hexagon
	'*',  # star
]

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
	"""Remove vertical spacing within paired rows of subplots.
	
	Args:
	    n_cols (int): Number of subplot columns; a scalar.
	    fig (matplotlib.figure.Figure): Scalar figure object whose axes are adjusted.
	    n_rows (int): Number of rows per repeated subplot group; a scalar.
	Returns:
	    matplotlib.figure.Figure: The modified input figure.
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
							linetype='-',
							fmt='',
							marker=None,
							markevery=None,
							markersize=5,
							elinewidth=1,
							capsize=3,
							set_xticks=True):
	"""Plot a 1-D series with optional lower/upper error bars.
	
	Args:
	    ax (matplotlib.axes.Axes): Scalar target axes object.
	    x (array-like): X coordinates with shape (m,).
	    mean (array-like): Central values with shape (m,).
	    lower (array-like, optional): Lower bounds with shape (m,).
	    upper (array-like, optional): Upper bounds with shape (m,).
	    label (str): Scalar legend label.
	    color (str): Scalar Matplotlib color specification.
	    linetype (str/tuple): Scalar Matplotlib line-style specification.
	    fmt (str): Scalar Matplotlib error-bar format string.
	    marker (str, optional): Scalar marker specification.
	    markevery (int/slice/sequence, optional): Scalar or 1-D marker-placement specification.
	    markersize (float): Scalar marker size.
	    elinewidth (float): Scalar error-bar line width.
	    capsize (float): Scalar error-bar cap size.
	    set_xticks (bool): Scalar flag for setting ticks to x.
	Returns:
	    None: Adds the series to ax in place.
	"""
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
				linestyle=linetype,
				capsize=capsize,
				marker=marker,
				markevery=markevery,
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
								   linetype='-',
								   marker=None,
								   markevery=None,
								   fmt='',
								   markersize=5,
								   elinewidth=1,
								   capsize=3,
								   set_xticks=True):
	"""Plot binomial means with Clopper-Pearson confidence intervals.
	
	Args:
	    ax (matplotlib.axes.Axes): Scalar target axes object.
	    x (array-like): X coordinates with shape (m,).
	    values (sequence): Length-m sequence of 1-D binary sample arrays.
	    alpha (float): Confidence-level tail probability; a scalar.
	    label (str): Scalar legend label.
	    color (str): Scalar Matplotlib color specification.
	    linetype (str/tuple): Scalar Matplotlib line-style specification.
	    marker (str, optional): Scalar marker specification.
	    markevery (int/slice/sequence, optional): Scalar or 1-D marker-placement specification.
	    fmt (str): Scalar Matplotlib error-bar format string.
	    markersize (float): Scalar marker size.
	    elinewidth (float): Scalar error-bar line width.
	    capsize (float): Scalar error-bar cap size.
	    set_xticks (bool): Scalar flag for setting ticks to x.
	Returns:
	    None: Adds the confidence-interval series to ax in place.
	"""
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
								   linetype=linetype,
								   marker=marker,
								   markevery=markevery,
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
	"""Plot binned central values and optional uncertainty intervals at bin centers.
	
	Args:
	    ax (matplotlib.axes.Axes): Scalar target axes object.
	    from_ (array): Lower bin edges with shape (B,).
	    to_ (array): Upper bin edges with shape (B,).
	    mean (array-like): Central bin values with shape (B,).
	    lower (array-like, optional): Lower bounds with shape (B,).
	    upper (array-like, optional): Upper bounds with shape (B,).
	    jitter (float): Scalar horizontal offset for bin centers.
	    color (str): Scalar Matplotlib color specification.
	    markersize (float): Scalar marker size.
	    label (str): Scalar legend label.
	Returns:
	    None: Adds the binned series to ax in place.
	"""
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
	"""Plot observed and predicted binned counts, optionally with their ratio panel.
	
	Args:
	    ax (matplotlib.axes.Axes): Scalar axes for count plots.
	    info (DotDic): Scalar result object containing fold runs and prediction helpers.
	    alpha (float): Bootstrap interval tail probability; a scalar.
	    tol (float): Numerical tolerance; a scalar.
	    lambda_ (float): Signal fraction shown in the label; a scalar.
	    binning (callable, optional): Function returning lower/upper bin arrays of shape (B,).
	    aggregate (int, optional): Scalar fold index to plot instead of aggregating all folds.
	    ax2 (matplotlib.axes.Axes, optional): Scalar axes for the observed/predicted ratio.
	    ax3 (matplotlib.axes.Axes, optional): Scalar auxiliary axes object; currently unused.
	    eps (float): Scalar plotting margin around [0, 1].
	Returns:
	    None: Draws plots on the supplied axes.
	"""
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
		predictions = [
			info.runs[aggregate].predict_counts(from_=from_, to_=to_)]
		methods = [info.runs[aggregate]]
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
		ax2.legend()

	ax.legend()


def cdfs(ax, df, labels, alpha, eps=1e-2):
	"""Plot empirical CDFs of p-value samples against the uniform CDF.
	
	Args:
	    ax (matplotlib.axes.Axes): Scalar target axes object.
	    df (sequence): Length-m sequence of 1-D p-value arrays.
	    labels (sequence): Length-m sequence of scalar legend strings.
	    alpha (float): Test level; a scalar (kept for a common plotting interface).
	    eps (float): Scalar axis margin around [0, 1].
	Returns:
	    None: Draws the CDFs on ax.
	"""
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
			linestyle=linetypes[i],
			marker=markers[i],
			markevery=8,
			alpha=1,
			label=labels[i])

	ax.set_ylabel('Cumulative probability')
	ax.set_xlabel('P-value')
	ax.legend()


def save_fig(cwd, path, fig, name):
	"""Save a Matplotlib figure as a PDF in the experiment results directory.
	
	Args:
	    cwd (str): Project working directory; a scalar string.
	    path (str): Relative results path; a scalar string.
	    fig (matplotlib.figure.Figure): Scalar figure object to save.
	    name (str): Output filename stem; a scalar string.
	Returns:
	    None: Saves and closes the figure.
	"""
	base_path = storage.get_path(cwd=cwd, path=path)
	filename = base_path + '{0}.pdf'.format(name)
	fig.savefig(fname=filename, bbox_inches='tight')
	print('\nSaved to {0}'.format(filename))
	plt.close(fig)