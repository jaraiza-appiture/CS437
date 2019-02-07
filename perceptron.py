import numpy as np

# Read a data file in csv format, separate into features and class arrays
def read_data():
   data = np.loadtxt(fname='and.csv', delimiter=',')
   #data = np.loadtxt(fname='or.csv', delimiter=',')
   #data = np.loadtxt(fname='xor.csv', delimiter=',')
   X = data[:, 1:]   # features
   y = data[:, 0]    # class
   return X, y


def compute_perceptron_output(features, weights, bias):
   sum = 0
   for i in range(len(features)):
      sum += weights[i] * features[i]
   sum += bias
   return sum


def perceptron(X, y):
   max_iterations = 20
   num_features = len(X[0])
   weights = [0.0] * num_features
   bias = 0.0
   total = len(X)

   for index in range(max_iterations):
      right = 0
      for i in range(len(X)):
         instance = X[i]
	 label = y[i]
         prediction = compute_perceptron_output(instance, weights, bias)
	 if label * prediction <= 0:
	    bias += label
	    for feature_num in range(len(instance)):
               weights[feature_num] += label * instance[feature_num]
         else:
	    right += 1
	 if index == 0:
	    print 'weights', weights, 'bias', bias
      if right == total:
         print 'Convergence!'
      print 'iteration', index, 'accuracy', float(right)/float(total), 'weights', weights


if __name__ == "__main__":
   X, y = read_data()
   perceptron(X, y)
