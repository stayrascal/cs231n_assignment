import numpy as np
import matplotlib.pyplot as plt

from cs231n.data_utils import load_CIFAR10

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_message():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)


def draw():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        # y_train : training labels array, 1,2,3,4...9,10
        # y_train == y : boolean array, [True, False]
        idxs = np.flatnonzero(y_train == y)
        # flatnonzero return indices that are non-zero in the flattened version of a
        # random choice samples_per_class elements from idexs array
        idxs = np.random.choice(idxs, samples_per_class, replace=False)

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(x_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def get_data():
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = 5000
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]

    num_test = 5000
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    return x_train, x_test, y_train, y_test, num_training, num_test

    # print(X_train.shape, X_test.shape)
    #
    # import pandas as pd
    # df = pd.DataFrame(X_train)
    # print(df.describe())
    # df = pd.DataFrame(X_test)
    # print(df.describe())


def main():
    x_train, x_test, y_train, y_test, num_training, num_test = get_data()
    classifier = get_classifier(x_train, y_train)

    dists = classifier.compute_distances_two_loops(x_test)
    print(dists.shape)

    # show_dists(dists)

    predict_labels(classifier, dists, y_test, num_test, 1)
    predict_labels(classifier, dists, y_test, num_test, 5)

    # calculate_time(classifier, dists, x_test)

    # divide_into_multiple_folds(x_train, y_train)


def predict_labels(classifier, dists, y_test, num_test, k=1):
    y_test_pred = classifier.predict_labels(dists, k)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


def calculate_time(classifier, dists, x_test):
    dists_one = classifier.compute_distances_one_loop(x_test)
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print('Difference was: %f' % (difference,))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')

    two_loop_time = time_function(classifier.compute_distances_two_loops, x_test)
    print('Two loop version took %f seconds' % two_loop_time)

    one_loop_time = time_function(classifier.compute_distances_one_loop, x_test)
    print('One loop version took %f seconds' % one_loop_time)

    no_loop_time = time_function(classifier.compute_distances_no_loops, x_test)
    print('No loop version took %f seconds' % no_loop_time)


def show_dists(dists):
    plt.imshow(dists, interpolation='none')
    plt.show()


def get_classifier(train_data, train_label):
    from cs231n.classifiers import KNearestNeighbor
    classifier = KNearestNeighbor()
    classifier.train(train_data, train_label)
    return classifier


def divide_into_multiple_folds(x_train, y_train):
    from cs231n.classifiers import KNearestNeighbor
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds)

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    for k in k_choices:
        k_to_accuracies[k] = []
        for i in range(num_folds):
            local_X = np.concatenate([X_train_folds[n] for n in range(num_folds) if n != i], axis=0)
            local_y = np.concatenate([y_train_folds[n] for n in range(num_folds) if n != i])
            local_classifier = KNearestNeighbor()
            local_classifier.train(local_X, local_y)
            local_dists = local_classifier.compute_distances_no_loops(X_train_folds[i])
            local_pred = local_classifier.predict_labels(local_dists, k=k)
            num_correct = np.sum(local_pred == y_train_folds[i])
            num_total = X_train_folds[i].shape[0]
            k_to_accuracies[k].append(float(num_correct) / num_total)

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
