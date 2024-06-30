from src.background.transform import exponential


def get_trans(args, min_):
	match args.target_data_id:
		case '3b' | '4b':
			return lambda X: exponential.trans(X=X,
											   rate=0.003,
											   base=min_,
											   scale=1)
		case 'WTagging':
			# In the WTagging dataset, the data is in the [0,1] scale.
			# Hence, no transformation in required.
			return lambda X: X

		case _:
			raise ValueError('Dataset not supported')
