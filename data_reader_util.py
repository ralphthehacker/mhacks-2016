from skdata.mnist.views import OfficialVectorClassification
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import time
import threading
# Idea: Understand what this does. What exactly is this queuing that we're doing and how this can be applied to the project.


# Base idea: Convert dataset into protobuf format. This enables the data to be easily read through a DataQueue
def serialize_Data():
    # Use a very simple dataset just to test
    data = OfficialVectorClassification()
    training_indexes = data.sel_idxs[:]

    # And shuffle the dataset. I don't know why this is necessary
    np.random.shuffle(training_indexes)


    # Finally, create a writer to make a TFRecords file out of the dataset
    writer = tf.python_io.TFRecordWriter("mnist.tfrecords") # MnistTFrecords is the path we're writing the records to

    # And iterate over all examples to create records
    for example_idx in tqdm(training_indexes):
        features = data.all_vectors[example_idx]
        label = data.all_labels[example_idx]  # Indexing the dataset for example and labels. In our case, we'll have notes

        # Construct the proto object that will be read in the future
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=features.astype("int64"))),
                }))
        # Write it as a string
        serialized_string = example.SerializeToString()

        # And write it to disk
        writer.write(serialized_string)


def read_serialized_data(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        # Create an example
        example = tf.train.Example()
        # And parse the information from the protorecord
        example.ParseFromString(serialized_example)
        # traverse the Example format to get data
        image = example.features.feature['image'].int64_list.value
        label = example.features.feature['label'].int64_list.value[0]
        # do something
        print label


def read_and_decode_single_example(filename):
    # First, build a queue with all the filenames
    filename_queue = tf.train.string_input_producer([filename],num_epochs=None)

    # And create a reader to go through the data.
    file_reader = tf.TFRecordReader()

    # Read one example
    _,serialized_example = file_reader.read(filename_queue)

    # And convert into an actual readable object
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([784], tf.int64)
        })
    label = features['label']
    images = features['image']

    return label, images

def test_read_and_decode(filename):
    # Read and decode the image
    label, image = read_and_decode_single_example(filename)

    # Initialize the session
    sess = tf.Session()

    # Initialize the queue and the main thread running the queue
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # grab examples back.
    # first example from file
    label_val_1, image_val_1 = sess.run([label, image])
    # second example from file
    label_val_2, image_val_2 = sess.run([label, image])

    print "EX 1"
    print label_val_1,image_val_1
    print "****************************"
    print "EX 2"
    print label_val_2, image_val_2


def test_batch_queue(filename):
    label,image = read_and_decode_single_example(filename)
    image = tf.cast(image,tf.float32)/255

    # Now, create the batch op
    images_batch, labels_batch = tf.train.shuffle_batch([image,label],batch_size= 250,capacity=2000,min_after_dequeue=1500)


    # Sample model
    w = tf.get_variable("w1", [28*28,10])
    y_pred = tf.matmul(images_batch,w)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

    # Create a monitoring variable
    loss_mean = tf.reduce_mean(loss)

    # And a training variable
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Start the batch queue
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # And do some training steps
    while True:
        _,loss_val = sess.run([train_op,loss_mean])

def data_iterator():
    batch_idx = True
    # TODO: Modify for song data
    while True:
            # shuffle labels and features
            idxs = np.arange(0, len(features))
            np.random.shuffle(idxs)
            shuf_features = features[idxs]
            shuf_labels = labels[idxs]
            for batch_idx in range(0, len(features), batch_size):
                images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.
                images_batch = images_batch.astype("float32")
                labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
                yield images_batch, labels_batch

class CustomRunner(object):
    ''' Manages the threads necessary to run the data queue'''

    def __init__(self):
        #TODO: Modify sizes so that they fit the music notes inputs
        self.dataX = tf.placeholder(dtype= tf.float32, shape=[None,28*28])
        self.dataY = tf.placeholder(dtype = tf.int64 , shape=[None, ])

        # Create the actual data queue
        self.queue = tf.RandomShuffleQueue(shapes=[[28*28],[]] ,
                                           dtypes=[tf.float32, tf.int64],
                                           capacity=2000,
                                           min_after_dequeue=1000)
        # And create a comp_graph operation to add data to the queue
        self.enqueue_op = self.queue.enqueue_many([self.dataX,self.dataY])

    def get_inputs(self, number):
        # again, it's necessary to modify the images/data part to fit our model
        images_batch, labels_batch = self.queue.dequeue_many(number)
        return images_batch, labels_batch

    def thread_main(self, sess):
        ''' Runs on another thread. Keeps adding more data to the queue'''
        for dataX, dataY in data_iterator():
            sess.run(self.enqueue_op, feed_dict = {self.dataX:dataX, self.dataY:dataY})


    def start_threads(self,sess,n_threads=1):
        ''' Starts the threads to run the queue'''
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads



def run_fast_queue(batch_size):
    # Doing anything with data on the CPU is generally a good idea.
    with tf.device("/cpu:0"):
        custom_runner = CustomRunner()
        images_batch, labels_batch = custom_runner.get_inputs(batch_size)

    # simple model
    w = tf.get_variable("w1", [28 * 28, 10])
    y_pred = tf.matmul(images_batch, w)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

    # for monitoring
    loss_mean = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
    init = tf.initialize_all_variables()
    sess.run(init)

    # start the tensorflow QueueRunner's
    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    custom_runner.start_threads(sess)

    while True:
        _, loss_val = sess.run([train_op, loss_mean])
        print loss_val








# load data entirely into memory
data = OfficialVectorClassification()
trIdx = data.sel_idxs[:]
features = data.all_vectors[trIdx]
labels = data.all_labels[trIdx]

batch_size = 128
run_fast_queue(batch_size)




# Then, make the DataQueue directly provide the data to the TF model




# Make a BS model and create the computational nodes


# Finally, send stuff from the queue into the loss function computation