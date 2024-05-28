from experiments import plot
from jax import numpy as np


def fits(args,
		 results,
		 path,
		 filename,
		 alpha,
		 binning=None,
		 eps=1e-2):
	keys = results.keys()
	n_cols = len(keys)
	height_ratios = [2, 1, 1.5]
	n_rows = 3
	height = np.sum(np.array(height_ratios)) / n_rows
	fig, axs = plot.plt.subplots(nrows=n_rows,
								 ncols=n_cols,
								 figsize=(10 * n_cols,
										  10 * height),
								 height_ratios=height_ratios,
								 sharex='row',
								 sharey='row')
	if n_cols == 1:
		axs = axs.reshape(-1, 1)

	for i, k in enumerate(keys):
		ax = axs[0, i]
		ax2 = axs[1, i]
		ax3 = axs[2, i]

		plot.hists(ax,
				   lambda_=0,
				   binning=binning,
				   ax2=ax2,
				   ax3=ax3,
				   info=results[k],
				   alpha=alpha,
				   tol=args.tol)

	fig = plot.tight_pairs(n_cols=n_cols, fig=fig)
	plot.save_fig(cwd=args.cwd,
				  path=path,
				  fig=fig,
				  name=filename)


def filtering(args,
			  lambdas,
			  quantiles,
			  results,
			  path,
			  alpha,
			  filename,
			  binning=None,
			  eps=1e-2):
	##################################################
	# Plot datasets with classifier filter
	##################################################
	n_cols = len(quantiles)
	height_ratios = [2, 1, 1.5]
	height_ratios = np.array([height_ratios] * len(lambdas)).reshape(-1)
	n_rows = len(lambdas) * 3
	height = np.sum(np.array(height_ratios)) / n_rows
	fig, axs = plot.plt.subplots(nrows=n_rows,
								 ncols=n_cols,
								 height_ratios=height_ratios,
								 figsize=(10 * n_cols,
										  10 * height),
								 sharex='row',
								 sharey='row')

	l = 0
	for lambda_ in lambdas:

		for q, quantile in enumerate(quantiles):

			ax = axs[l, q]
			ax2 = axs[l + 1, q]
			ax3 = axs[l + 2, q]

			if l == 0:
				ax.set_title(
					'Filtering {0}% of the observations'.format(
						int(quantile * 100)))

			plot.hists(ax,
					   lambda_=lambda_,
					   binning=binning,
					   info=results[lambda_][quantile],
					   ax2=ax2,
					   ax3=ax3,
					   alpha=alpha,
					   tol=args.tol)

		l += 3

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
		tests = [results[lambda_][quantile].tests for quantile in quantiles]

		plot.binary_series_with_uncertainty(
			ax,
			x=np.array(quantiles) * 100,
			values=tests,
			color=plot.colors[i],
			label='$\lambda$={0}'.format(lambda_),
			alpha=alpha)

	ax.legend()


def power_per_classifier(args, path, classifiers, labels, results, lambdas,
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
		axs[c].set_title(labels[c], fontsize=20)

	plot.save_fig(cwd=args.cwd, path=path, fig=fig, name='power')


def power_per_quantile(args, path, results, lambdas, quantiles):
	plot_ = lambda classifier, quantile: plot.cdfs(
		ax=ax,
		df=[results[classifier][lambda_][quantile].stats for lambda_ in
			lambdas],
		labels=['$\lambda=${0}'.format(lambda_) for lambda_ in lambdas],
		alpha=args.alpha)

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
