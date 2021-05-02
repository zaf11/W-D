import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

outpath = '../../../data/train.tfrecords'

class WideAndDeep(tf.estimator.Estimator):
    def __init__(self, random_seed=2021):
        self._init_graph_()
        self.random_seed = random_seed

    def _init_graph_(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

        pass


if __name__=="__main__":
    dataset = tf.data.TFRecordDataset(outpath)
