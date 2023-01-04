import keras.optimizers
import tensorflow as tf
import numpy as np
import streamlit as st
from keras.callbacks import History
import plotly.express as px

####  SET WEB CONFIGURATION AND TITLES   #########################################################
st.set_page_config (page_title='Playing with keras - linear regression', layout="wide")
st.write ("""
    # Simple Keras Regression  
    ### Multivariate regression with keras to play with different model params.  
    Data has been generated with random numpy for x,y,z and a target function:--->   targets = 2*x - 5*y + z + noise  
    No dataset division (train, test, validation) and no accuracy measures in this simple version.
    """)

####  READ DATA AND SHOW IT   #######################################################################
training_data = np.load('data_all.npz')
col1,col2 = st.columns([1,5])

with col1:
    st.write('### Inputs')
    st.write(training_data['inputs'])
with col2:
    st.write('### Targets')
    st.write(training_data['targets'])

####  SOME FUNCTIONS and constants      ############################################################

# a list of the activation functions that can be used
act_fun_list = ['linear', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']

# a list of optimizers
optimizer_list = ['SGD', 'RMSprop', 'Adam','AdamW','Adadelta','Adamax','Adafactor','Nadam','Ftrl' ]

# a list regression losses
losses_list = ['mean_squared_error','mean_absolute_error', 'mean_absolute_percentage_error',
               'mean_squared_logarithmic_error', 'cosine_similarity']



# print model results
def model_weights(model):
    model.layers [0].get_weights ()
    weights = model.layers [0].get_weights () [0]
    bias = model.layers [0].get_weights () [1]
    st.write ('### Results: weights and bias ')
    st.write ('weights --> ' , str (weights))
    # st.write ('a --> ' + str (weights [0]) + '; b --> ' + str (weights [1]) + '; c --> ' + str (weights [2]))
    st.write ('bias --> ', str (bias))




####  SPLIT SCREEN TO COMPARE MODELS       ############################################################

col3,col4 = st.columns([1,1])

with col3:

    # model epochs
    st.write('#### Select number of epochs - A ')
    num_epochs_A = st.slider ('### epochs A', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write('#### Select the activation function - A ')
    activation_f_A = st.radio ("### Choose activation function A", act_fun_list)

    model_A = tf.keras.Sequential([tf.keras.layers.Dense(
        units=1,
        activation=activation_f_A,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        # kernel_initializer=tf.random_uniform_initializer (minval=-0.1, maxval=0.1),
        # bias_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,)])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write('#### Select an optimizer - A ')
    optimizer_sel = st.radio ("### Choose an optimizer A", optimizer_list )
    learn_rate_A = st.slider ('Learning rate of optimizers A: ', min_value=0.0001, max_value=0.5000, step=0.0001, value=0.02, format="%f")

    optimizer_A = f"tf.keras.optimizers.{optimizer_sel}(learning_rate={learn_rate_A})"
    optimizer_A = eval(optimizer_A)


    st.write('#### Select regression losses  - A ')
    losses_A = st.radio ("### Choose regression losses A", losses_list)

    model_A.compile (optimizer=optimizer_A, loss=losses_A)

    # train the model
    history_A = History()
    model_A.fit (training_data['inputs'], training_data['targets'], epochs=num_epochs_A ,verbose=0, callbacks=[history_A])   #verbose 2 para ver todo

    # print model results
    model_weights(model_A)

    # print loss function graph
    loss = history_A.history ['loss']
    xs = range (num_epochs_A)
    fig = px.line (x=xs, y=loss,
                   title="Loss function by epochs",
                   labels=dict (x="Number of epochs case A", y="Loss value case A"))
    st.plotly_chart (fig, use_container_width=True)


with col4:

    # model epochs
    st.write('#### Select number of epochs - B ')
    num_epochs_B = st.slider ('### epochs B', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write('#### Select the activation function - B ')
    activation_f_B = st.radio ("### Choose activation function B", act_fun_list)
    model_B = tf.keras.Sequential([tf.keras.layers.Dense(
        units=1,
        activation=activation_f_B,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        # kernel_initializer=tf.random_uniform_initializer (minval=-0.1, maxval=0.1),
        # bias_initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1),
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,)])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write('#### Select an optimizer - B ')
    optimizer_sel_B = st.radio ("### Choose an optimizer B", optimizer_list)
    learn_rate_B = st.slider ('Learning rate of optimizers B: ', min_value=0.0001, max_value=0.5000, step=0.0001, value=0.02, format="%f")

    optimizer_B = f"tf.keras.optimizers.{optimizer_sel_B}(learning_rate={learn_rate_B})"
    optimizer_B = eval (optimizer_B)

    st.write('#### Select regression losses  - B ')
    losses_B = st.radio ("### Choose regression losses B", losses_list)

    model_B.compile (optimizer=optimizer_B, loss=losses_B)

    # train the model
    history_B = History ()
    model_B.fit (training_data ['inputs'], training_data ['targets'], epochs=num_epochs_B, verbose=0,
               callbacks=[history_B])  # verbose 2 para ver todo

    # print model results
    model_weights(model_B)

    # print loss function graph
    loss_B = history_B.history ['loss']
    xs_B = range (num_epochs_B)
    fig_B = px.line (x=xs_B, y=loss_B,
                   title="Loss function by epochs",
                   labels=dict (x="Number of epochs case B", y="Loss value case B"))
    st.plotly_chart (fig_B, use_container_width=True)

