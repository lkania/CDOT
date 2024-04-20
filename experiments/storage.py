from pathlib import Path


def get_path(cwd, path):
	path_ = '{0}/results/{1}/'.format(cwd, path)
	Path(path_).mkdir(parents=True, exist_ok=True)
	return path_


import pickle


def save_obj(cwd, path, obj, name):
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('\nSaved to {0}'.format(filename))


def load_obj(cwd, path, name):
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	with open(filename, 'rb') as handle:
		obj = pickle.load(handle)
	return obj
