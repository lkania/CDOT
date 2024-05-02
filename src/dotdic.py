import copy


class DotDic(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

	# Note that we define hashes manually
	def __hash__(self):
		return self.hash

	def copy(self):
		return copy.deepcopy(self)
