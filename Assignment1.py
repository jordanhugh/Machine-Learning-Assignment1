import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("week1-2 .csv",comment='#')
x=np.array(df.iloc[:,0]); x=x.reshape(-1, 1)
y=np.array(df.iloc[:,1]); y=y.reshape(-1, 1)

mean_x = np.mean(x)
std_x = np.std(x)
x = (x - mean_x)/std_x
mean_y = np.mean(y)
std_y = np.std(y)
y = (y - mean_y)/std_y

m = np.size(x)
learning_rates = [0, 0.001, 0.01, 0.1]
iterations = 200
cost_funcs = []
epsilons = []
for learning_rate in learning_rates:
    cost_func = []
    epsilon = [0,0]
    for i in range(iterations):
        h = epsilon[0] + epsilon[1] * x
        cost_func.append(1/m * np.sum((h - y) ** 2))
        delta = [0]*2
        delta[0] = -(2 * learning_rate)/m * np.sum(h - y)
        delta[1] = -(2 * learning_rate)/m * np.sum((h - y) * x)
        epsilon[0] = epsilon[0] + delta[0]
        epsilon[1] = epsilon[1] + delta[1]
    cost_funcs.append(cost_func)
    epsilons.append(epsilon)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.set_title('Cost Function', fontsize=14)
ax.set_xlabel('Iterations', fontsize=12)
ax.set_ylabel('J (θ)', fontsize=12)
for i in range(len(learning_rates)):
    ax.plot(range(iterations), cost_funcs[i])
labels = ['Baseline']
for learning_rate in learning_rates[1:]:
    labels.append('α='+str(learning_rate))
ax.legend(labels, loc='upper right')
plt.savefig('cost_functions')
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax = ax.flatten()
fig.tight_layout(pad=4.0)
for itr, learning_rate in enumerate(learning_rates):
    if itr == 0:
        ax[itr].set_title('α = ' + str(learning_rate) + ' (Baseline)', fontsize=14)
    else:
        ax[itr].set_title('α = ' + str(learning_rate), fontsize=14)
    ax[itr].set_xlabel('X (Normalised)', fontsize=12)
    ax[itr].set_ylabel('Y (Normalised)', fontsize=12)
    ax[itr].scatter(x, y, linewidth=1, color='C' + str(itr), alpha=0.4)
    ax[itr].plot(x, epsilons[itr][0] + epsilons[itr][1] * x, linewidth=3, color='C' + str(itr))
    ax[itr].legend(['Predictions', 'Training Data'], loc='upper right')
plt.savefig('lr_with_learning_rate')
plt.show()

regr = LinearRegression()
regr.fit(x, y)
y_pred = regr.predict(x)

print('Parameter Values')
print('%-6s %-12s %-12s' % ('α', 'e0', 'e1'))
for itr, learning_rate in enumerate(learning_rates):
    print('%-6.3f %-12.8f %-12.8f' % (learning_rate, epsilons[itr][0], epsilons[itr][1]))
print()
    
print('Cost Function')
print('%-6s %-12s' % ('α', 'J(θ)'))
for itr, learning_rate in enumerate(learning_rates):
    print('%-6.3f %-12.8f' % (learning_rate, cost_funcs[itr][-1]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.set_title("Sklearn Linear Regression Model", fontsize=14)
ax.set_xlabel("X (Normalised)", fontsize=12)
ax.set_ylabel("y (Normalised)", fontsize=12)
plt.scatter(x, y, linewidth=1, alpha=0.4)
plt.plot(x, y_pred, linewidth=3)
plt.legend(["Predictions","Training Data"], loc='upper right') 
plt.savefig('sklearn_linear_regression_model')
plt.show()
    
print('Sklearn Parameter Values:')
print('%-24s %-16s %-12s' % ('Model','Intercept', 'Coefficient'))
print('%-24s %-16.8e %-12.8f' % ('Gradient Descent(α=' + str(learning_rates[-1]) + ')', epsilons[-1][0], epsilons[-1][1]))
print('%-24s %-16.8e %-12.8f' % ('Sklearn', regr.intercept_, regr.coef_))