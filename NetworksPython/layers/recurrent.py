class RecLayer():
    def __init__(self, **kwargs):
        self.sName = kwargs['sName']

    def evolve(*args, **kwargs):
        pass

    def __str__(self):
        return '{} object: "{}"'.format(self.__class__.__name__, self.sName)

    def __repr__(self):
        return self.__str__()