import numpy as np

X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], 
              [1, 1, 0], [1, 1, 1]), dtype=float)

y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

xPredicted = np.array(([1,0,1]), dtype=float)
X = X / np.amax(X, axis=0)
xPredicted = xPredicted / np.amax(xPredicted, axis=0)



class NeuralNet(object):
    def __init__(self):
        self.input_size = 3  
        self.hidden_size = 4  
        self.output_size = 1

        # W1 is a matrix with the dimensions (input, hidden) -> (3,4)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)

        # W2 is a matrix with dimensions (hidden, output) -> (4,1)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self,x):
      return x * (1 - x)

    def feed_forward(self, X):
      self.z = np.dot(X, self.W1)
      self.z2 = self.sigmoid(self.z)
      self.z3 = np.dot(self.z2, self.W2)
      o = self.sigmoid(self.z3)
      return o

    def back_prop(self, X, y, o):
      self.o_error = y - o # actual - predicted (output error)
      self.o_delta = self.o_error * self.sigmoid_deriv(o)
      self.z2_error = self.o_delta.dot(self.W2.T) #How much hidden layer weights 
#contributed to output error

      self.z2_delta = self.z2_error * self.sigmoid_deriv(self.z2)

      self.W1 += X.T.dot(self.z2_delta)
      self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
      o = self.feed_forward(X)
      self.back_prop(X, y, o)

    def predictOutput(self, xPredicted):
      print("Predicted XOR output data based on trained weights: \n")
      print(f"Data (X1-X3): \n{xPredicted}\n")
      print(f"Output (Y1): \n{self.feed_forward(xPredicted)}")


    


neuralnet = NeuralNet()
trainingEpochs = 2000

print(f"Network Input: \n{X}\n")
print(f"Expected Output of XOR Gate Neural Network: \n{y}\n")

Loss = 0

for i in range(trainingEpochs):
  if i % 50 == 0:
    print(f"Epoch: {i}")
  Loss = np.mean(np.square(y - neuralnet.feed_forward(X)))
  neuralnet.trainNetwork(X, y)
  
print("Epoch: 2000\n")
print (f"Accuracy: \n{(1 - Loss) * 100}%\n")
neuralnet.predictOutput(xPredicted)


print(f"\nW1 Matrix: \n{neuralnet.W1}\n")
print(f"\nW2 Matrix: \n{neuralnet.W2}")



