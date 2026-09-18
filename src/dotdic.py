import copy


class DotDic(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __getattr__(self, name):
		"""Read a dictionary entry using attribute syntax.
		
		Args:
		    self: DotDic mapping object.
		    name: String scalar naming the requested key.
		Returns:
		    Stored object for that key; its type/dimension depend on the value."""
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

	# Note that we define hashes manually
	def __hash__(self):
		"""Return the explicitly stored hash value for this parameter dictionary.
		
		Args:
		    self: DotDic mapping expected to contain a scalar/hashable `hash` entry.
		Returns:
		    Hashable scalar stored in self.hash."""
		return self.hash

	def copy(self):
		"""Create an independent deep copy of this parameter dictionary.
		
		Args:
		    self: DotDic mapping, with values of arbitrary supported types/dimensions.
		Returns:
		    DotDic/deep-copied mapping with the same keys and value dimensions."""
		return copy.deepcopy(self)
