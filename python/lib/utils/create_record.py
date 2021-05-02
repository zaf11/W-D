import os
import timeit
from itertools import islice
import tensorflow as tf

inpath = '../../../data/data.csv'
outpath = '../../../data/train.tfrecords'

def create_record():
    start_time = timeit.default_timer()
    writer = tf.io.TFRecordWriter(outpath)

    f = open(inpath, "r")
    for line in islice(f, 1, None):
        data = line.split(',')
        label = int(data[0])
        features = [float(i) if i!="" else float(0) for i in data[1:]]

        example = tf.train.Example(features=tf.train.Features(feature={
            "label" : tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
            "features" : tf.train.Feature(float_list = tf.train.FloatList(value=features)),
        }))
    writer.write(example.SerializeToString())
    writer.close()

    end_time = timeit.default_timer()
    print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))


if __name__ == "__main__":
    create_record()