import sys

import os.path
from pathlib import Path
import cloudpickle as pickle


def get_path(cwd, path):
	"""Create and return the directory used for experiment results.
	
	Args:
	    cwd (str): Project working directory; a scalar string.
	    path (str): Relative result subdirectory; a scalar string.
	Returns:
	    str: Scalar absolute/combined results-directory path.
	"""
	path_ = '{0}/results/{1}/'.format(cwd, path)
	Path(path_).mkdir(parents=True, exist_ok=True)
	return path_


def save_obj(cwd, path, obj, name):
	"""Serialize a Python object to a pickle file in the results directory.
	
	Args:
	    cwd (str): Project working directory; a scalar string.
	    path (str): Relative result subdirectory; a scalar string.
	    obj (object): Scalar Python object/container to serialize; dimensions depend on the object.
	    name (str): Pickle filename stem; a scalar string.
	Returns:
	    None: Writes the object to disk.
	"""
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	with open(filename, 'wb') as handle:
		pickle.dump(obj, handle)
	print('\nSaved to {0}'.format(filename))


def exists(cwd, path, name):
	"""Check whether a cached pickle result exists.
	
	Args:
	    cwd (str): Project working directory; a scalar string.
	    path (str): Relative result subdirectory; a scalar string.
	    name (str): Pickle filename stem; a scalar string.
	Returns:
	    bool: Scalar existence indicator.
	"""
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	return os.path.isfile(filename)


def load_obj(cwd, path, name):
	"""Load a cached Python object from a pickle file.
	
	Args:
	    cwd (str): Project working directory; a scalar string.
	    path (str): Relative result subdirectory; a scalar string.
	    name (str): Pickle filename stem; a scalar string.
	Returns:
	    object: Deserialized Python object; type and dimensions match the saved object.
	"""
	filename = get_path(cwd=cwd, path=path) + '{0}.pickle'.format(name)
	print('\nLoading {0}'.format(filename))
	with open(filename, 'rb') as handle:
		obj = pickle.load(handle)
	return obj
