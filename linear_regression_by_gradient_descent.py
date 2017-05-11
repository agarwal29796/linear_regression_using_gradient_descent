from numpy import *
import matplotlib.pyplot as plt 


def compute_error_for_given_points(b, m, points):
    totalError = 0 
    for i in range(0 , len(points)):
        x = points[i , 0]
        y = points[i , 1]
        totalError += (y - (m*x + b))**2
    return totalError/float(len(points))



def step_gradient(b_current,  m_current , points , learningRate):
   #GRadient_descent
   b_gradient = 0 
   m_gradient = 0 
   N = float(len(points))
   for i in range(0, len(points)):
       x = points[i , 0]
       y = points[i , 1]
       b_gradient += -(2/N)*(y-((m_current*x)+b_current))
       m_gradient += -(2/N)*x*(y-((m_current*x)+b_current))
   new_b = b_current - (learningRate * b_gradient)
   new_m = m_current - (learningRate*m_gradient)
   return [new_b , new_m]

     
def gradient_descent_runner(points, starting_b , starting_m , learning_rate ,num_iterations):
    b = starting_b 
    m = starting_m
    approximated_values = [(b, m)]  
    
    for i in range(num_iterations):
        b,m = step_gradient(b ,m, array(points), learning_rate )
        approximated_values.append((b,m))
        
    return  approximated_values 

def run():
    points = genfromtxt('data.csv', delimiter = ',')
    #hyper parameters
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0 
    initial_m = 0 
    num_iterations = 1000
    approximated_values = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    #print(points)
#        plt.plot((approximated_values[i])[0] ,(approximated_values[i])[0])
    for i in range(len(points)):
        plt.scatter((points[i])[0] , (points[i])[1])
    final_intercept = (approximated_values[len(approximated_values)-1])[0]
    final_slope = (approximated_values[len(approximated_values)-1])[1]
    # some dummy data
    x = [0 , 100]
    y = [final_slope*x[i] + final_intercept for i in range(len(x))]
    plt.plot(x,y)
    plt.show()
if __name__ == '__main__':
    run()