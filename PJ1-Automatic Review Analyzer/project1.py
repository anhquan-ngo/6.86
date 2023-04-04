from string import punctuation, digits
import numpy as np
import random
# Part I

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    y_pred = label * (np.dot(feature_vector, theta) + theta_0)
    loss_single = max(0, 1 - y_pred)
    return loss_single
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    total_loss = 0
    for row, label in zip(feature_matrix, labels):
        total_loss += hinge_loss_single(row,label,theta,theta_0)
    return total_loss/labels.size
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    if label * (np.dot(current_theta,feature_vector)+current_theta_0) <= 0:
        current_theta += label*feature_vector
        current_theta_0 += label
    return (current_theta,current_theta_0)
    raise NotImplementedError


def perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],theta, theta_0)
    return (theta, theta_0)
    raise NotImplementedError


def average_perceptron(feature_matrix, labels, T):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    sum_theta = np.zeros(feature_matrix.shape[1])
    sum_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i],labels[i],theta, theta_0)
            sum_theta += theta
            sum_theta_0 += theta_0
    return (sum_theta/(feature_matrix.shape[0]*T), sum_theta_0/(feature_matrix.shape[0]*T))
    raise NotImplementedError


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    if label * (np.dot(current_theta, feature_vector) + current_theta_0) <= 1:
        current_theta = (1-eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 += eta*label
    else:
        current_theta = (1-eta*L)*current_theta
    return (current_theta, current_theta_0)
    raise NotImplementedError


def pegasos(feature_matrix, labels, T, L):
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    count = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            count += 1
            eta = 1/(count**0.5)
            theta,theta_0 = pegasos_single_step_update(feature_matrix[i],labels[i],L, eta, theta, theta_0)
    return (theta,theta_0)
    raise NotImplementedError

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    y_pred_lst = []
    for row in feature_matrix:
        y_pred = np.dot(theta, row) + theta_0
        if y_pred > 0:
            y_pred_lst.append(1)
        else:
            y_pred_lst.append(-1)
    y_pred_arr = np.array(y_pred_lst)
    return y_pred_arr
    raise NotImplementedError


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    y_pred_train = classify(train_feature_matrix, theta, theta_0)
    acc_train = accuracy(y_pred_train, train_labels)
    y_pred_val = classify(val_feature_matrix, theta, theta_0)
    acc_val = accuracy(y_pred_val, val_labels)
    return (acc_train, acc_val)
    raise NotImplementedError

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    with open('stopwords.txt', 'r') as file:
        file_content = file.read()
        stopwords_lst = file_content.split()
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if (word not in dictionary) and (word not in stopwords_lst):
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_list.count(word)
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
