import numpy
import scipy.special
import matplotlib.pyplot


class neuralNetwork:

    def __init__(self,input_nodes,hidden1_nodes,hidden2_nodes,output_nodes,learning_rate):
        self.inodes = input_nodes
        self.h1nodes = hidden1_nodes
        self.h2nodes = hidden2_nodes
        self.onodes = output_nodes
        self.lrate = learning_rate

        self.wih1 = numpy.random.normal(0.0,pow(self.h1nodes,-0.5),(self.h1nodes,self.inodes))
        self.wh1h2 = numpy.random.normal(0.0,pow(self.h2nodes,-0.5),(self.h2nodes,self.h1nodes))

        self.wh2o = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.h2nodes))

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self,inputs_list,targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)


        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden2_errors = numpy.dot(self.wh2o.T, output_errors)
        hidden1_errors = numpy.dot(self.wh1h2.T, hidden2_errors)

        self.wh2o += self.lrate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden2_outputs))

        self.wh1h2 += self.lrate * numpy.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), numpy.transpose(hidden1_outputs))
        self.wih1 += self.lrate * numpy.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):

        inputs = numpy.array(inputs_list, ndmin=2).T


        hidden1_inputs = numpy.dot(self.wih1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = numpy.dot(self.wh1h2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)


        final_inputs = numpy.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)


        print(final_outputs)
        return final_outputs




input_nodes = 784
hidden1_nodes = 500
hidden2_nodes = 100

output_nodes = 10

learning_rate = 0.3

n = neuralNetwork(input_nodes,hidden1_nodes,hidden2_nodes,output_nodes,learning_rate)

training_data_file = open("mnist_dataset/mnist_train_100.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')

    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    #print(record)
    n.train(inputs,targets)
    pass

test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

test_all_values = test_data_list[2].split(',')
print(test_all_values[0])
#print((numpy.asfarray(test_all_values[1:])/ 255.0 * 0.99) + 0.01)
n.query((numpy.asfarray(test_all_values[1:])/ 255.0 * 0.99) + 0.01)
