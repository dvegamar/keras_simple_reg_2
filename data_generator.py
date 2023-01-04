import numpy as np


observations = 1000
x = np.random.uniform(low=-10, high=10, size=(observations,1))
y = np.random.uniform(low=-10, high=10, size=(observations,1))
z = np.random.uniform(low=-10, high=10, size=(observations,1))
noise = np.random.uniform(low=-1, high=1, size=(observations,1))
var_inputs = np.column_stack((x,y,z))   #creates a 1000x3 matriz
targets = 2*x - 5*y + z + noise

np.savez('data_all', inputs= var_inputs, targets=targets)

# print var_inputs dimensions

print ('inputs dimensions ' ,var_inputs.shape)
print ('output dimensions ' ,targets.shape)


# let´s print the array - this step is informative
array = np.column_stack((x,y,z,noise,targets))
np.set_printoptions(suppress=True) #skips scientific notation
print (np.around(array,4))