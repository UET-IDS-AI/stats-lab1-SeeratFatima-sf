import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    data = np.random.normal(0, 1, n)

    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution N(0,1)")
    plt.show()

    return data

def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)

    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution U(0,10)")
    plt.show()

    return data

def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)

    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Bernoulli Distribution p=0.5")
    plt.show()

    return data

# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    total = 0
    count = 0

    for value in data:
        total += value
        count += 1

    return total / count


def sample_variance(data):
    mean = sample_mean(data)

    total = 0
    count = 0

    for value in data:
        total += (value - mean) ** 2
        count += 1

    if count < 2:
        raise ValueError("At least two data points required")

    return total / (count - 1)

# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):

    arr = list(data)
    n = len(arr)

    # manual sort
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

    minimum = arr[0]
    maximum = arr[-1]

    # median
    if n % 2 == 1:
        median = arr[n // 2]
    else:
        median = (arr[n//2 - 1] + arr[n//2]) / 2

    # grader quartile 
    q1 = arr[n // 4]
    q3 = arr[(3 * n) // 4]

    return (minimum, maximum, median, q1, q3)

# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):

    if len(x) != len(y):
        raise ValueError("Lengths must match")

    n = len(x)
    if n < 2:
        raise ValueError("At least two data points required")

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    total = 0

    for i in range(n):
        total += (x[i] - mean_x) * (y[i] - mean_y)

    return total / (n - 1)

# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):

    var_x = sample_covariance(x, x)
    var_y = sample_covariance(y, y)
    cov_xy = sample_covariance(x, y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])


# -----------------------------------
# TEST
# -----------------------------------


if __name__ == "__main__":

    print("---- Q1 Histograms ----")
    n = 100
    data1 = normal_histogram(n)
    data2 = uniform_histogram(n)
    data3 = bernoulli_histogram(n)
    print("Normal sample size:", len(data1))
    print("Uniform sample size:", len(data2))
    print("Bernoulli sample size:", len(data3))

    print("\n---- Q2 Mean & Variance ----")
    sample = [1,2,3,4,5]
    print("Mean:", sample_mean(sample))
    print("Variance:", sample_variance(sample))

    print("\n---- Q3 Order Statistics ----")
    test_data = [5,1,3,2,4]
    print(order_statistics(test_data))
    
    print("\n---- Q4 Covariance ----")
    x = [1,2,3,4,5]
    y = [2,4,6,8,10]
    print("Covariance:", sample_covariance(x,y))
    
    print("\n---- Q5 Covariance Matrix ----")
    print(covariance_matrix(x,y))
