### Chapter 3 | Introduction to neural prediction

##  A simple neural network making a prediction with a single input
# 1. An empty network
weight = 0.1
def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 2. Inserting one input datapoint
number_of_toes = [8.5, 9.5, 10, 9]
input = number_of_toes[0]
prediction = neural_network(input, weight)
print(prediction) # 0.85

# 3. Multiplying input by weight
def neural_network(input, weight):
    prediction = input * weight
    return prediction

# 4. Depositing the prediction
number_of_toes = [8.5, 9.5, 10, 9]
input = number_of_toes[0]
prediction = neural_network(input, weight)


##  Making a prediction with multiple inputs
# 1. An empty network with multiple inputs
weights = [0.1, 0.2, 0]
def neural_network(input, weights):
    prediction = w_sum(input, weights)
    return prediction

# 2. Inserting one input datapoint
toes = [8.5, 9.5, 9.9, 9.0] # average number of toes per player
wlrec = [0.65, 0.8, 0.8, 0.9] # win/loss record of games won (percent)
nfans = [1.2, 1.3, 0.5, 1.0] # number of fans (millions)

input = [toes[0], wlrec[0], nfans[0]]

prediction = neural_network(input, weights)

# 3. Performing a weighted sum (dot product) of inputs
def w_sum(a, b):
    assert(len(a) == len(b))

    output = 0

    for i in range(len(a)):
        output += (a[i] * b[i])

    return output

def neural_network(input, weights):
    prediction = w_sum(input, weights)
    return prediction

# 4. Depositing the prediction
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0],wlrec[0],nfans[0]]

prediction = neural_network(input, weights)

print(prediction) # 0.98


# Challenge: Vector math
# elementwise multiplication
def elementwise_multiplication(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    output = []
    for i in range(len(vec_a)):
        output[i] = vec_a[i] * vec_b[i]
    return output

# elementwise addition
def elementwise_addition(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    output = []
    for i in range(len(vec_a)):
        output[i] = vec_a[i] + vec_b[i]
    return output

# vector sum
def vector_sum(vec_a):
    output = 0
    for i in range(len(vec_a)):
        output += vec_a[i]
    return output

# vector average
def vector_average(vec_a):
    result = vector_sum(vec_a)
    output = result / len(vec_a)
    return output

# perform a dot product
def perform_dot_product(vec_a, vec_b):
    assert(len(vec_a) == len(vec_b))
    result_multiplication = elementwise_multiplication(vec_a, vec_b)
    result_weighted_sum =  vector_sum(result_multiplication)
    return result_weighted_sum


# Making a prediction with multiple outputs
# 1. An empty network with multiple outputs
weights = [0.3, 0.2, 0.9]
def neural_network(input, weights):
    prediction = ele_mul(input, weights)
    return prediction

# 2. Inserting one input datapoint
wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
prediction = neural_network(input, weights)

# 3. Performing elementwise multiplication
def ele_mul(number, vector):
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

def neural_network(input, weights):
    prediction = ele_mul(input, weights)
    return prediction

# 4. Depositing predictions
wlrec = [0.65, 0.8, 0.8, 0.9]
input = wlrec[0]
prediction = neural_network(input, weights)
print(prediction) # [0.195, 0.13, 0.585]


# Predicting with multiple inputs and multiple outputs
# 1. An empty network with multiple inputs and outputs
weights = [
    # toes # win # fans
    [0.1, 0.1, -0.3], # hurt?
    [0.1, 0.2, 0.0], # win?
    [0.0, 1.3, 0.1] # sad?
]

# 2. Inserting one input datapoint
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0], wlrec[0], nfans[0]]
prediction = neural_network(input, weights)

# 3. For each output, performing a weighted sum of inputs
def w_sum(a, b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output

# vector matrix multiplication
def vect_mat_mul(vector, matrix):
    assert(len(vector) == len(matrix))
    output = [0, 0, 0]

    for i in range(len(vector)):
        output[i] = w_sum(vector, matrix[i])
    
    return output

def neural_network(input, weights):
    prediction = vect_mat_mul(input, weights)
    return prediction

# 4. Depositing predictions
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0], wlrec[0], nfans[0]]
prediction = neural_network(input, weights)
print(prediction) # [0.555, 0.98, 0.965]


# Neural networks can be stacked
# 1. An empty network with multiple inputs and outputs
ih_wgt = [ 
    # toes % win # fans
    [0.1, 0.2, -0.1], # hid[0]
    [-0.1,0.1, 0.9], # hid[1]
    [0.1, 0.4, 0.1]  # hid[2]
]

hp_wgt = [ 
    #hid[0] hid[1] hid[2]
    [0.3, 1.1, -0.3], # hurt?
    [0.1, 0.2, 0.0], # win?
    [0.0, 1.3, 0.1] 
] # sad?

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0])
    prediction = vect_mat_mul(hid, weights[1])
    return prediction

# 2. Predicting the hidden layer
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65,0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]

prediction = neural_network(input, weights)

def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0])
    prediction = vect_mat_mul(hid, weights[1])
    return prediction

# 3. Predicting the output layer (and depositing the prediction)
def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0])
    prediction = vect_mat_mul(hid, weights[1])
    return prediction

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0], wlrec[0], nfans[0]]

prediction = neural_network(input, weights)
print(prediction) # [0.214, 0.145, 0.507]


# NumPy version
import numpy as np

ih_wgt = np.array([
    # toes % win # fans
    [0.1, 0.2, -0.1], # hid[0]
    [-0.1,0.1, 0.9], # hid[1]
    [0.1, 0.4, 0.1]]).T # hid[2]

hp_wgt = np.array([
    # hid[0] hid[1] hid[2]
    [0.3, 1.1, -0.3], # hurt?
    [0.1, 0.2, 0.0], # win?
    [0.0, 1.3, 0.1] ]).T # sad?

weights = [ih_wgt, hp_wgt]

def neural_network(input, weights):
    hid = input.dot(weights[0])
    prediction = hid.dot(weights[1])
    return prediction

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65,0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])

prediction = neural_network(input, weights)
print(prediction) # [0.214, 0.145, 0.507]


# A quick primer on NumPy
import numpy as np

a = np.array([0,1,2,3]) # vector
b = np.array([4,5,6,7]) # vector
c = np.array([
    [0,1,2,3],
    [4,5,6,7]]) # matrix
d = np.zeros((2,4)) # 2x4 matrix of 0s
e = np.random.rand(2,5) # 2x5 matrix of random numbers between 0 and 1 

print(a) # [0 1 2 3]
print(b) # [4 5 6 7]
print(c) # [[0 1 2 3] [4 5 6 7]]
print(d) # [[ 0. 0. 0. 0.] [ 0. 0. 0. 0.]]
print(e) # [[ 0.22717119 0.39712632 0.0627734 0.08431724 0.53469141] [ 0.09675954 0.99012254 0.45922775 0.3273326 0.28617742]]

print(a * 0.1) # multiplies every number in vector a by 0.1
print(c * 0.2) # multiplies every number in matrix c by 0.2
print(a * b) # multiplies elementwise between a and b (columns paired)
print(a * b * 0.2) # multiplies elementwise, then multiplies by 0.2
print(a * c) # multiplies elementwise on every row of matrix c because c has the same number of columns as a
print(a * e) # throws an error “Value Error: operands could not be broadcast together with...” because a and e don’t have the same number of columns

a = np.zeros((1,4))
b = np.zeros((4,3))
c = a.dot(b)
print(c.shape) # (1,3)
