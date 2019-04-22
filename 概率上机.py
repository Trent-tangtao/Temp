"""

Homework  1，2
        by 唐涛  2019.4.3

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# problem one
random50 = np.random.randn(50)
random1000 = np.random.randn(1000)
random10000 = np.random.randn(10000)
# print(random50)
# print(random1000)
# print(random10000)


# problem two
random01 = np.random.normal(loc=0, scale=1, size=5000)
random41 = np.random.normal(loc=4, scale=1, size=5000)
random04 = np.random.normal(loc=0, scale=4, size=5000)


def show_Normal_distributio(x, mu, sigma, num_bins=50):
    mu, sigma, num_bins = mu, sigma, num_bins
    n, bins, patches = plt.hist(x, num_bins, density=True, rwidth=0.8, alpha=0.5)
    y = st.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('Histogram of normal distribution: $\mu = %d$, $\sigma=%d$' % (mu, sigma))
    plt.subplots_adjust(left=0.15)
    plt.show()


show_Normal_distributio(random01, 0, 1)
show_Normal_distributio(random04, 0, 4)
show_Normal_distributio(random41, 4, 1)

