import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.loadtxt("data2.txt", delimiter=",")
x1 = np.array(data[0:47,0:1])
x2 = np.array(data[0:47,1:2])
y = np.array(data[0:47,2:3])
normilized_x1 = preprocessing.minmax_scale(x1) #size of house
normilized_x2 = preprocessing.minmax_scale(x2) #number of badroom
normilized_y = preprocessing.minmax_scale(y) #price
#print(y)
def gradient_descent(x1,x2,y):
    theta_one = 0
    theta_two = 0
    theta_not = 0
    steps = 1000;
    n= len(x1)
    alpha = 0.7
    last_cost = 175
    for i in range(steps):
        y_predicted = theta_not + theta_one * x1 + theta_two * x2
        cost = (1 / (2*n) ) * sum([val ** 2 for val in (y - y_predicted)])
        #print(cost)
        theta_one_derivative = - (2/n) * sum(x1*(y-y_predicted))
        theta_not_derivative = - (2/n) * sum(y - y_predicted)
        theta_two_derivative = - (2/n) * sum(x2*(y-y_predicted))
        theta_one = theta_one - alpha * theta_one_derivative
        theta_not = theta_not - alpha * theta_not_derivative
        theta_two = theta_two - alpha * theta_two_derivative
        print("theta_one {}     theta_two{}     theta_not {}     steps {}   cost {}".format(theta_one, theta_two, theta_not, i, cost))
        #print(last_cost - cost)


        if(last_cost - cost)> (10**-3):
            plt.figure("cost Vs steps")
            plt.ylabel("cost")
            plt.xlabel("steps")
            plt.scatter(i,cost ,color="green")
            last_cost = cost
        else:
            h = theta_one * x1 + theta_two* x2 + theta_not
            plt.figure("Data Hypothesis")
            ax = plt.axes(projection='3d')
            ax.scatter3D(x1,x2,h, cmap='Greens');
            break

    print("Learning Parameter : ", alpha)


gradient_descent(normilized_x1,normilized_x2,normilized_y)
plt.show()



