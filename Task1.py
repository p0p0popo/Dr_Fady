import numpy as np


class TrainMachine:
    def __init__(self):
        np.random.seed(1)
        self.init_weights = 2 * np.random.random((3,1))-1
        print("initial starting weights")
        print(self.init_weights)
        # input dataset
        self.training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])
        # output dataset
        self.training_outputs = np.array([[0.0,1.0,1.0,0]]).T

    def sigmoid(self,x):
        return 1 / ( 1 + np.exp(-x) )

    def train(self):
        output = np.dot(self.training_inputs , self.init_weights)
        self.activiation_output = self.sigmoid(output)
        return self.activiation_output

machine1 = TrainMachine()
print(machine1.train())
