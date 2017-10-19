import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 100
n_nodes_hl3 = 50

#No of output classes
n_classes = 10
#No of training examples
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

print(x)
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    #rectified linear
    l1 = tf.nn.relu(l1)
    #print(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    #rectified linear
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    #rectified linear
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    print("main done",output)
    return output


def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range (hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict= {x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)

filenames = ['/home/achintha/Academics/3.png']
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

image = tf.image.decode_jpeg(value, channels=1)
resized_image = tf.image.resize_images(image, [28, 28])
print(resized_image)
reshaped_image = tf.reshape(resized_image, [tf.shape(resized_image)[0],784])
print(reshaped_image)

final = neural_network_model(reshaped_image)
print("dim final", final)
#final = neural_network_model(x)
#print(final)
model = tf.global_variables_initializer()

#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
#print(final.eval())

with tf.Session() as sess:
    sess.run(model)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("hello")
    print(sess.run(final))
    print("okay")
