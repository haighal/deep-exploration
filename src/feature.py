import numpy as np

class FeatureExtractor(object):
    """
    Base feature extractor.
    """
    def __init__(self, **kwargs):
        pass

    def get_feature(self, **kwargs):
        pass


class MountainCarIdentityFeature(FeatureExtractor):
    """
    Returns the current state vector (x, x_dot).
    """
    def __init__(self):
        self.dimension = 2

    def get_feature(self, observation_history):
        return observation_history[-1][0]