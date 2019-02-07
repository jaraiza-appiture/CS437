from numpy import log2, loadtxt
from random import shuffle
# from google.colab import drive
# drive.mount('/content/gdrive')

def load_data():
  data = loadtxt(fname='/content/gdrive/My Drive/har.csv',
                 delimiter=',')
  X = data[:,:-1] #features
  y = data[:,-1]  #labels
  return X,y

# My entropy function
def entropy(S,classes):
   total = len(S)
   e=0.0
   for uni_val in classes:
      n = len([row[-1] for row in S if row[-1] == uni_val])
      if n > 0:
        p_i = float(n) / total
        e+= -p_i*log2(p_i)
        
   return e

# My gain info funciton for discrete values
def gainDiscrete(S,A):
   attri_vals = list(set([row[A] for row in S]))
   total = len(S)
   igain = entropy(S)
   for uni_val in attri_vals:
      S_v = [row for row in S if row[A] == uni_val]
      n = len(S_v)
      igain-= (n/total) * entropy(S_v)
   return igain

# My gain info function for continous values
def gainContinous(S_entropy,groups,classes):
   total = float(sum([len(group) for group in groups]))
   igain = S_entropy
   for group in groups:
      n = len(group)
      if n > 0:
        igain-= (n/total) * entropy(group,classes)
   return igain

# Calculate the Gini index for a subset of the dataset
def gini_index(groups, classes):
   # count all samples at split point
   num_instances = float(sum([len(group) for group in groups]))

   gini = 0.0 # sum weighted Gini index for each group
   for group in groups:
      size = float(len(group))
      if size == 0: # avoid divide by zero
         continue
      score = 0.0
      # score the group based on the score for each class
      for class_val in classes:
         p = [row[-1] for row in group].count(class_val) / size
         score += p * p
      # weight the group score by its relative size
      gini += (1.0 - score) * (size / num_instances)
   return gini

# Create child splits for a node or make a leaf node
def split(node, max_depth, depth):
   left, right = node['groups']
   del(node['groups'])
   # check for a no split
   if not left or not right:
      node['left'] = node['right'] = create_leaf(left + right)
      return
   # check for max depth
   if depth >= max_depth:
      node['left'], node['right'] = create_leaf(left), create_leaf(right)
      return

   node['left'] = select_attribute(left)
   split(node['left'], max_depth, depth+1)
   node['right'] = select_attribute(right)
   split(node['right'], max_depth, depth+1)

# split the dataset based on an attribute and attribute value
def test_split(index, value, dataset):
   left, right = list(), list()
   for row in dataset:
      if row[index] < value:
         left.append(row)
      else:
         right.append(row)
   return left, right

# Select the best split point for a dataset
def select_attribute(dataset):
   class_values = list(set(row[-1] for row in dataset))
   dataset_entropy = entropy(dataset,class_values)
   b_index, b_value, b_score, b_groups = 999, 999, -1.0,None
   for index in range(len(dataset[0])-1):
      for row in dataset:
         groups = test_split(index, row[index], dataset)
         gainVal = gainContinous(dataset_entropy,groups,class_values)
         if gainVal > b_score:
            b_index, b_value, b_score, b_groups = index, row[index], gainVal, groups
   return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a leaf node class value
def create_leaf(group):
   outcomes = [row[-1] for row in group]
   return max(set(outcomes), key=outcomes.count)

# Build a decision tree
def build_tree(train, max_depth):
   root = select_attribute(train)
   print("Root Node: %s"%(root))
   split(root, max_depth, 1)
   return root

# Print a decision tree
def print_tree(node, depth=0):
   if depth == 0:
      print('Tree:')
   if isinstance(node, dict):
      print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
      print_tree(node['left'], depth+1)
      print_tree(node['right'], depth+1)
   else:
      print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
   if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
         return predict(node['left'], row)
      else:
         return node['left']
   else:
      if isinstance(node['right'], dict):
         return predict(node['right'], row)
      else:
         return node['right']

def train_test_split(data,split_ratio):
   shuffle(data)
   indexes = int(len(data)*split_ratio)
   train, test = data[:indexes], data[indexes:]
   return train, test

if __name__ == "__main__":
#    dataset = [[2.771244718,1.784783929,0], [1.728571309,1.169761413,0],
#               [3.678319846,2.812813570,0], [3.961043357,2.619950320,0],
#               [2.999208922,2.209014212,0], [7.497545867,3.162953546,1],
#               [9.00220326, 3.339047188,1], [7.444542326,0.476683375,1],
#               [10.12493903,3.234550982,1], [6.642287351,3.319983761,1]]
  X,y = load_data()
  
  dataset = X.tolist()
  yl = y.tolist()
  for i in range(len(dataset)):
    dataset[i].append(yl[i])

  train,test = train_test_split(dataset,2/3)
  tree = build_tree(train, 3)
  print_tree(tree)

  total_predictions = len(test)
  correct_predictions = 0
  for row in test:
     prediction = predict(tree, row)
     if prediction == row[-1]:
        correct_predictions+=1
  print("Correct / Total Predictions: %.3f"%(correct_predictions/total_predictions))
