import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.loadtxt("data1.txt", delimiter=",")
x = np.array(data[0:97,0:1])
y = np.array(data[0:97,1:2])

normilized_x = preprocessing.minmax_scale(x)
normilized_y = preprocessing.minmax_scale(y)

#print(normilized_x)
#print(x)

def gradient_descent(x,y):
    theta_one = 0
    theta_not = 0
    steps = 1000;
    n= len(x)
    alpha = 0.94
    last_cost = 1
    for i in range(steps):
        y_predicted = theta_one * x + theta_not
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        #print(cost)
        theta_one_derivative = - (2/n) * sum(x*(y-y_predicted))
        theta_not_derivative = - (2 / n) * sum(y - y_predicted)
        theta_one = theta_one - alpha * theta_one_derivative
        theta_not = theta_not - alpha * theta_not_derivative
        print("theta_one {}     theta_not {}     steps {}   cost {}".format(theta_one, theta_not, i, cost))
        #print(last_cost - cost)


        if(last_cost - cost)> (10**-3):
            plt.figure("cost Vs iteration")
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.scatter(i,cost ,color="green")
            last_cost = cost
        else:
            h = theta_one * x + theta_not
            plt.figure("Data Hypothesis")
            plt.scatter(x,y)
            plt.scatter(x,h,marker="+")
            break

    print("Learning Parameter : ", alpha)


gradient_descent(normilized_x,normilized_y)
plt.show()



