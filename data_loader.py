import pickle,numpy as np
from random import shuffle
import random
import json, os, sys,datetime
import matplotlib.pyplot as plt

class LoadModRecData:
    def __init__(self, datafile, trainRatio, validateRatio, testRatio):
        ''' init

            .. note::

                calls :func:`loadData` and :func:`split`
        '''

        # check python version
        self.python_version_3 = False
        if sys.version_info >= (3, 0):
            self.python_version_3 = True

        print("Loading Data...")

        self.signalData, self.oneHotLabels, self.signalLabels = self.loadData(datafile)
        self.train_idx, self.val_idx, self.test_idx = self.split(trainRatio, validateRatio, testRatio)

        print("Done.\n")

    def loadData(self, fname):
        '''  Load dataset from pickled file '''

        # load data from files
        with open(fname, 'rb') as f:
            if self.python_version_3:
                self.dataCube = pickle.load(f, encoding='latin-1')
                dataCubeKeyIndices = list(zip(*self.dataCube))
            else:
                self.dataCube = pickle.load(f)
                dataCubeKeyIndices = zip(*self.dataCube)
        # get all mod types
        self.modTypes = np.unique(dataCubeKeyIndices[0])
        print(self.modTypes);
        # get all SNR values
        self.snrValues = np.unique(dataCubeKeyIndices[1])

        # create one-hot vectors for each mod type
        oneHotArrays = np.eye(len(self.modTypes), dtype=int)
        # Count Number of examples
        print("Counting Number of Examples in Dataset...")
        number_of_examples = 0
        for modType in self.modTypes:
            for snrValue in self.snrValues:
                number_of_examples = number_of_examples + len(self.dataCube[modType, snrValue])

        print('Number of Examples in Dataset: ' + str(number_of_examples))

        # pre-allocate arrays
        signalData = [None] * number_of_examples
        oneHotLabels = [None] * number_of_examples
        signalLabels = [None] * number_of_examples

        # for each mod type ... for each snr value ... add to signalData, signalLabels, and create one-Hot vectors
        example_index = 0
        one_hot_index = 0
        self.instance_shape = None

        for modType in self.modTypes:
            print("[Modulation Dataset] Adding Collects for: " + str(modType))
            for snrValue in self.snrValues:

                # get data for key,value
                collect = self.dataCube[modType, snrValue]

                for instance in collect:
                    signalData[example_index] = instance
                    signalLabels[example_index] = (modType, snrValue)
                    oneHotLabels[example_index] = oneHotArrays[one_hot_index]
                    example_index += 1

                    if self.instance_shape is None:
                        self.instance_shape = np.shape(instance)

            one_hot_index += 1  # keep track of iteration for one hot vector generation

        # convert to np.arrays
        print("Converting to numpy arrays...")
        signalData = np.asarray(signalData)
        oneHotLabels = np.asarray(oneHotLabels)
        signalLabels = np.asarray(signalLabels)

        # Shuffle data
        print("Shuffling Data...")
        """ signalData_shuffled, signalLabels_shuffled, oneHotLabels_shuffled """
        # Randomly shuffle data, use predictable seed
        np.random.seed(2017)
        shuffle_indices = np.random.permutation(np.arange(len(signalLabels)))
        signalData_shuffled = signalData[shuffle_indices]
        signalLabels_shuffled = signalLabels[shuffle_indices]
        oneHotLabels_shuffled = oneHotLabels[shuffle_indices]

        return signalData_shuffled, oneHotLabels_shuffled, signalLabels_shuffled

    def split(self, trainRatio, validateRatio, testRatio):
        '''  split dataset into train, validation, and test '''

        # Split data into train/validate/test via indexing
        print("Splitting Data...")

        # Determine how many samples go into each set
        [num_sigs, num_samples] = np.shape(self.oneHotLabels)
        num_train = np.int(np.floor(num_sigs * trainRatio))
        num_val = np.int(np.floor(num_sigs * validateRatio))
        num_test = np.int(num_sigs - num_train - num_val)

        print(
        'Train Size: ' + str(num_train) + ' Validation Size: ' + str(num_val) + ' Test Size: ' + str(num_test))
        # Generate a random permutation of the sample indicies
        rand_perm = np.random.permutation(num_sigs)

        # Asssign Indicies to sets
        train_idx = rand_perm[0:num_train]
        val_idx = rand_perm[num_train:num_train + num_val]
        #test_idx = rand_perm[num_sigs-num_test:]
        test_idx = rand_perm[num_train + num_val:]

        return train_idx, val_idx, test_idx

    def get_indicies_withSNRthrehsold(self, indicies, snrThreshold_lowBound, snrThreshold_upperBound):
        '''  get batch from indicies with SNR threshold  - (inclusive >=lowerBound, <= upperBound) '''

        filteredIndicies = []
        i = 0
        for snrValue in self.signalLabels[indicies][:, 1]:
            if int(snrValue) >= int(snrThreshold_lowBound) and int(snrValue) <= int(snrThreshold_upperBound):
                filteredIndicies.extend([indicies[i]])
            i += 1

        return filteredIndicies

    def get_batch_from_indicies(self, indicies):
        '''  get batch from indicies  '''

        batch_x = self.signalData[indicies]
        batch_y = self.oneHotLabels[indicies]
        batch_y_labels = self.signalLabels[indicies]

        # return the batch
        return zip(batch_x, batch_y, batch_y_labels)

    def get_random_batch(self, index_list, batch_size):
        ''' get batch of specific size from dataset '''

        rand_pool = np.random.choice(np.shape(index_list)[0], size=batch_size, replace=False)
        rand_idx = index_list[rand_pool[0:batch_size]]

        # use the indices to get the data for x and y
        batch_x = self.signalData[rand_idx]
        batch_y = self.oneHotLabels[rand_idx]
        batch_y_labels = self.signalLabels[rand_idx]

        # return the batch
        return zip(batch_x, batch_y, batch_y_labels)

    def batch_iter(self, data_indicies, batch_size, num_epochs, use_shuffle=True):
        """
        provide generator for iteration of the training indicies created during initialization

        # iteration - one batch_size from data
        # epoch - one pass through all of the data
        """

        data_size = len(data_indicies)
        num_batches_per_epoch = int(len(data_indicies) / batch_size)

        for epoch in range(num_epochs):
            # Shuffle the indices at each epoch
            if use_shuffle:
                shuffle(data_indicies)

            # loop across all example data one batch at a time
            for batch_num in range(num_batches_per_epoch):
                # determine start index
                start_index = batch_num * batch_size

                # determine end index, min of iteration vs end of data
                end_index = min((batch_num + 1) * batch_size, data_size)

                # get indices of the instances in the batch
                indices_of_instances_in_batch = data_indicies[start_index:end_index]

                # use the indices to get the data for x and y
                batch_x = self.signalData[indices_of_instances_in_batch]
                batch_y = self.oneHotLabels[indices_of_instances_in_batch]
                batch_y_labels = self.signalLabels[indices_of_instances_in_batch]

                # return a training batch
                yield zip(batch_x, batch_y, batch_y_labels)