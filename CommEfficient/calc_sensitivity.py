import numpy as np
import sys
if __name__ == "__main__":
    sigma = float(sys.argv[1])
    eps = 2
    delta = 1e-5
    sensitivity = np.sqrt((sigma ** 2 * eps ** 2)/(2 * np.log(1.25 / delta)))
    print(sensitivity)
