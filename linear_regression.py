import pandas as pd 
import matplotlib.pyplot as plt


data = pd.read_csv('Salary_Data.csv')



# how much you are off from the actual result
def loss_function(m, b, points):
    total_error =0 
    for i in range(len(points)):
        x = points.iloc[i].YearsExperience
        x = points.iloc[i].Salary
        #actual y - what we thought the y point should be - squared 
        total_error += (y - (m * x + b)) ** 2 
    total_error / float(len(points))


# pass current values, points and learning rate
def gradient_descent(m_now, b_now, points, L_rate):
    m_gradient = 0
    b_gradient = 0 

    n = len(points)

    for i in range (n):
        x = points.iloc[i].YearsExperience
        y = points.iloc[i].Salary

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        m_gradient += -(2/n) * (y - (m_now * x + b_now))

    m - m_now - m_gradient * L_rate
    b - b_now - b_gradient * L_rate
    return m, b

m = 0 
b = 0
L_rate = 0.001
epoch = 1000

for i in range(epoch):
    if i % 50 == 0:
        print(f'Epoch: {i}')
    m, b = gradient_descent(m, b, data, L_rate)

print(m, b)
plt.scatter(data.YearsExperience, data.Salary, color='black')
plt.plot(list(range(1, 11)), [m * x + b for x in range(1, 11)], color='red')
plt.show()


