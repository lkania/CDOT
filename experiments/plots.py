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
	fig, axs = plot.plt.subplots(nrows=2,
								 ncols=n_cols,
								 figsize=(10 * n_cols,
										  10),
								 height_ratios=[2, 1],
								 sharex='all',
								 sharey='row')

	axs[0, 0].set_ylabel('Counts ($\lambda=0$)')
	axs[1, 0].set_ylabel('Obs / Pred')

	for i, k in enumerate(keys):
		test = results[k].test
		ax = axs[0, i]
		ax2 = axs[1, i]
		ax.set_xlim([0 - eps, 1 + eps])
		ax2.set_xlim([0 - eps, 1 + eps])

		runs = results[test.name].runs

		plot.hists(ax,
				   binning=binning,
				   ax2=ax2,
				   methods=runs,
				   alpha=alpha)

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
	fig, axs = plot.plt.subplots(nrows=len(lambdas) * 2,
								 ncols=n_cols,
								 height_ratios=np.array(
									 [[2, 1]] * len(lambdas)).reshape(-1),
								 figsize=(10 * n_cols,
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

			tests = results[lambda_][quantile].runs

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
				np.array(results[lambda_][quantile].pvalues <= results[lambda_][
					quantile].test.threshold,
						 dtype=np.int32)
				for quantile in quantiles],
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
		df=[results[classifier][lambda_][quantile].pvalues for lambda_ in
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
