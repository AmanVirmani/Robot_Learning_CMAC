import math
import random
import numpy as np
import matplotlib.pyplot as plt
import headers.defaults as defaults
from headers.cmac import CMAC


def plot_results(train_error_list_discrete, test_error_list_discrete, train_error_list_cont,
                     test_error_list_cont, turnaround_time_list_discrete, turnaround_time_list_cont,
                     x_label):
    """ plot cmac performance graphs
    :param train_error_list_discrete: list of training errors for discrete CMAC
    :param test_error_list_discrete: list of testing errors for discrete CMAC
    :param train_error_list_cont: list of training errors for continuous CMAC
    :param test_error_list_cont: list of testing errors for continuous CMAC
    :param turnaround_time_list_discrete: list of time of convergence for discrete CMAC
    :param turnaround_time_list_cont: list of time of convergence for continuous CMAC
    :param x_label: x-label for graphs
    :return: nothing
    """

    if x_label is str('GeneralizationFactor'):
        range_values = range(1, defaults.max_generalization_factor + 1)
        value = ' Input Space Size = ' + str(defaults.plot_input_space_size)
    else:
        range_values = range(defaults.min_input_space_size, defaults.max_input_space_size,
                             defaults.input_space_step_size)
        value = ' Generalization Factor = ' + str(defaults.plot_generalization_factor)

    plt.figure(figsize=(20, 11))
    # Plot training errors graph
    plt.subplot(221)
    plt.plot(range_values, train_error_list_discrete, 'b', label='Discrete CMAC')
    plt.plot(range_values, train_error_list_cont, 'r', label='Continuous CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Train Error')
    plt.legend(loc='upper right', shadow=True)
    plt.title('CMAC ' + ' \n ' + value + ' \n ' + x_label + ' Vs Training Error')
    # Plot testing errors graph
    plt.subplot(222)
    plt.plot(range_values, test_error_list_discrete, 'b', label='Discrete CMAC')
    plt.plot(range_values, test_error_list_cont, 'r', label='Continuous CMAC')
    plt.xlabel(x_label)
    plt.ylabel('Test Error')
    plt.legend(loc='upper right', shadow=True)
    plt.title('CMAC '+ '\n ' + x_label + ' Vs Testing Error')
    # Plot convergence time graph for discrete CMAC
    plt.subplot(223)
    plt.plot(range_values, turnaround_time_list_discrete)
    plt.xlabel(x_label)
    plt.ylabel('Convergence Time')
    plt.title('Discrete CMAC')
    # Plot convergence time graph for continuous CMAC
    plt.subplot(224)
    plt.plot(range_values, turnaround_time_list_cont)
    plt.xlabel(x_label)
    plt.ylabel('Convergence Time')
    plt.title('Continuous CMAC')
    plt.subplots_adjust(0.1, 0.08, 0.89, 0.89, 0.2, 0.35)
    plt.show()


def gen_data(func, input_space_size=defaults.plot_input_space_size, input_min=0,
                     input_max=360, dataset_split_factor=0.7):
    """ generate dataset for CMAC with 70:30 train to test ratio
    :param func: mathematical function to test and train CMAC, by default consider sinusoid
    :param input_space_size: total no. of points for entire dataset
    :param input_min: default value 0
    :param input_max: default value 360
    :param dataset_split_factor: specify the ratio to split training and testing data points
    :return: list of parameters for the dataset
    """
    # Define step size to get 100 data points
    step_size = (input_max - input_min) / float(input_space_size)
    # Get 100 x and y for the sine function
    input_space = [math.radians(step_size * (i + 1)) for i in range(0, input_space_size)]
    output_space = [func(input_space[i]) for i in range(0, input_space_size)]
    # Get size of training and testing dataset
    training_set_size = int(input_space_size * dataset_split_factor)
    testing_set_size = input_space_size - training_set_size
    # Generate a zeros array for training dataset
    training_set_input = np.zeros(training_set_size).tolist()
    training_set_output = np.zeros(training_set_size).tolist()
    train_set_global_indices = np.zeros(training_set_size).tolist()
    # Generate a zeros array for testing dataset
    testing_set_input = np.zeros(testing_set_size).tolist()
    testing_set_true_output = np.zeros(testing_set_size).tolist()
    test_set_global_indices = np.zeros(testing_set_size).tolist()

    count = 0
    randomized_range_values = [x for x in range(0, input_space_size)]
    random.shuffle(randomized_range_values)

    input_step_size = (math.radians(input_max) - math.radians(input_min)) / float(input_space_size)
    # Add data to empty arrays
    for i in randomized_range_values:
        if count < training_set_size:
            training_set_input[count] = input_space[i]
            training_set_output[count] = output_space[i]
            train_set_global_indices[count] = i
        else:
            testing_set_input[count - training_set_size] = input_space[i] + (random.randrange(0, 10) * 0.01)
            output_space[i] = func(testing_set_input[count - training_set_size])
            testing_set_true_output[count - training_set_size] = output_space[i]
            test_set_global_indices[count - training_set_size] = i
        # Increment count to add data at the correct index of the array
        count = count + 1

    return [input_space, output_space, training_set_input, training_set_output,
            train_set_global_indices, testing_set_input, testing_set_true_output,
            test_set_global_indices, input_step_size]


def main(func):
    """ run CMAC for a given function,which can be provided using math library
    :param func: mathematical function to test and train CMAC
    :return: nothing
    """

    # Generate dataset for the sinusoid
    data = gen_data(func)
    # Run CMAC for various generalization factors
    discrete_cmac = [CMAC(i, data, 'DISCRETE') for i in
                     range(defaults.min_generalization_factor, defaults.max_generalization_factor + 1)]
    continuous_cmac = [CMAC(i, data, 'CONTINUOUS') for i in
                       range(defaults.min_generalization_factor, defaults.max_generalization_factor + 1)]


#    discrete_cmac = [CMAC(i, data, 100, 'DISCRETE') for i in
#                     range(defaults.min_generalization_factor, defaults.max_generalization_factor + 1)]
#    continuous_cmac = [CMAC(i, data, 100, 'CONTINUOUS') for i in
#                       range(defaults.min_generalization_factor, defaults.max_generalization_factor + 1)]

    train_error_list_discrete = [0 for i in range(0, defaults.max_generalization_factor)]
    test_error_list_discrete = [0 for i in range(0, defaults.max_generalization_factor)]

    train_error_list_cont = [0 for i in range(0, defaults.max_generalization_factor)]
    test_error_list_cont = [0 for i in range(0, defaults.max_generalization_factor)]

    turnaround_time_list_discrete = [100 for i in range(0, defaults.max_generalization_factor)]
    turnaround_time_list_cont = [100 for i in range(0, defaults.max_generalization_factor)]

    best_discrete_cmac = -1
    best_continuous_cmac = -1

    lowest_testing_error_discrete = 1000
    lowest_testing_error_continuous = 1000

    # Plotting a cmac with no generalization effect
    print(' \n  Plot Generalization Factor = ' + str(defaults.plot_generalization_factor + 1) + ' with Errors \n')
    continuous_cmac[defaults.plot_generalization_factor].execute()
    continuous_cmac[defaults.plot_generalization_factor].plot_graphs()

    discrete_cmac[defaults.plot_generalization_factor].execute()
    discrete_cmac[defaults.plot_generalization_factor].plot_graphs()

    # Plotting a cmac with the best generalization effect
    print(' \n Generalization Factor Variance - CMAC Performance \n ')
    for i in range(0, defaults.max_generalization_factor):
        train_error_list_discrete[i], test_error_list_discrete[i] = discrete_cmac[i].execute()
        train_error_list_cont[i], test_error_list_cont[i] = continuous_cmac[i].execute()

        print('Generalization Factor - ' + str(i + 1) + ' Continuous Testing Error - ' + str(
            round(test_error_list_cont[i], 3)) + ' Continuous Convergence Time - ' + str(
            round(continuous_cmac[i].convergence_time, 2)) + ' Discrete Testing Error - ' + str(
            round(test_error_list_discrete[i], 3)))

        turnaround_time_list_discrete[i] = discrete_cmac[i].convergence_time
        turnaround_time_list_cont[i] = continuous_cmac[i].convergence_time

        if test_error_list_discrete[i] < lowest_testing_error_discrete:
            lowest_testing_error_discrete = test_error_list_discrete[i]
            best_discrete_cmac = i

        if test_error_list_cont[i] < lowest_testing_error_continuous:
            lowest_testing_error_continuous = test_error_list_cont[i]
            best_continuous_cmac = i

    if best_discrete_cmac is not -1:
        discrete_cmac[best_discrete_cmac].plot_graphs()
    else:
        print("Error in finding best Discrete CMAC")

    if best_continuous_cmac is not -1:
        continuous_cmac[best_continuous_cmac].plot_graphs()
    else:
        print("Error in finding best Continuous CMAC")

    # Plot performance graphs with increasing generalization factor
    plot_results(train_error_list_discrete, test_error_list_discrete, train_error_list_cont,
                     test_error_list_cont, turnaround_time_list_discrete, turnaround_time_list_cont,
                     'GeneralizationFactor')


if __name__ == '__main__':
    # Run CMAC for cosine func
    main(math.cos)
