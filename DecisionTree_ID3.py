#Decision tree implementation without using inbuilt libraries
from random import randrange
from csv import reader
import math
import sys


# Load a .data file with names and data
def load_csv(loadfile):
    file = open(loadfile, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Split a dataset into a train and test set 80:20
def train_test_split(dataset, split=0.80):
    train = list()
    train_size = split * len(dataset)
    dataset_duplicate = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_duplicate))
        train.append(dataset_duplicate.pop(index))
    return train, dataset_duplicate


# Split a dataset into k folds for cross validation (specified n_folds=5)
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_duplicate = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_duplicate))
            fold.append(dataset_duplicate.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a single split
def evaluate_algorithm_single(dataset, algorithm, *args):
    train_set, test_set = train_test_split(dataset, 0.80)
    scores = list()
    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    return scores


# Evaluate algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    number_of_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class to select the node
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / number_of_instances)
    return gini


# Calculate the Entropy for a split dataset
def entropy(groups, classes, b_score):
    # count all samples at split point
    number_of_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    ent = 0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
        if p > 0:
            score = (p * math.log(p, 2))
        # weight the group score by its relative size i.e Entrpy gain
        ent -= (score * (size / number_of_instances))
    return ent


# Select the best split point for a dataset
def get_split(dataset, split_parameter):
    if split_parameter == 'entropy':  # this is invoked for parameter entropy
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 1, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                ent = entropy(groups, class_values, b_score)
                if ent < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], ent, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}
    elif split_parameter == 'gini':  # this is invoked for parameter gini
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 99999, 99999, 1, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value that is completing the tree with final values
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, split_parameter)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, split_parameter)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size, split_parameter):
    root = get_split(train, split_parameter)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[ATTRIBUTE[%s] = %.50s]' % ((depth * '\t', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


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


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size, split_parameter):
    tree = build_tree(train, max_depth, min_size, split_parameter)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions)


# Datasets used to load and prepare data (iris, house-votes-84)
value = sys.argv[2]
loadfile = value  # Provide the .data loadfile on which you want to test
dataset = load_csv(loadfile)
# Tree model creation on training set
n_folds = 5
max_depth = 3
min_size = 1
split_parameter = 'entropy'  # 'entrpy'/'gini'
train_set, test_set = train_test_split(dataset, 0.80)
tree = build_tree(train_set, max_depth, min_size, split_parameter)
print('Dictionary Representation of tree on training set')
print('  ')
print(tree)
print('  ')
print('Attributes ')
print(dataset[0])
print('Textual format of Decision tree')
print_tree(tree)
scores_1 = evaluate_algorithm_single(dataset, decision_tree, max_depth, min_size, split_parameter)
print('  ')
print('Implementing Single Split')
print('Scores: %s' % scores_1)
print('Accuracy: %.3f%%' % (sum(scores_1) / float(len(scores_1))))
# Calculating scores for k cross validation by setting n_folds=5 value
print('  ')
print('Implementing k-cross validation')
scores_2 = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size, split_parameter)
print('Scores: %s' % scores_2)
print('Mean Accuracy: %.3f%%' % (sum(scores_2) / float(len(scores_2))))