from collections import defaultdict


class DotDic(defaultdict):
    def __init__(self):
        super(DotDic, self).__init__(DotDic)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value
