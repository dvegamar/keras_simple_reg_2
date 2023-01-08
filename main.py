import keras.optimizers
import tensorflow as tf
import numpy as np
import streamlit as st
from keras.callbacks import History
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


##################################################################################################
####  SET WEB CONFIGURATION AND TITLES   #########################################################
##################################################################################################

st.set_page_config (page_title='Playing with keras - linear regression', layout="wide")
st.write ("""
    # Simple Keras Regression  
    ### Multivariate regression with keras to play with different model params.  
    Data has been generated with random numpy for x,y,z and a target function:--->   targets = 2*x - 5*y + z + noise  
    
    Second approach with data split in train, validation and test sets, plus a final accuracy measure.
    """)

##################################################################################################
####  READ DATA AND SHOW IT   ####################################################################
##################################################################################################

training_data = np.load ('data_all.npz')
col1, col2 = st.columns ([1, 5])

with col1:
    st.write ('### Inputs')
    st.write (training_data ['inputs'])
with col2:
    st.write ('### Targets')
    st.write (training_data ['targets'])

########  SPLIT THE DATASET   ####################################################################

size = len (training_data ['inputs'])
size_train = int (0.8 * size)
size_val = int (0.1 * size)
size_test = int (0.1 * size)

data_train = training_data ['inputs'] [0:size_train, ]
data_val = training_data ['inputs'] [size_train:size_train + size_val, ]
data_test = training_data ['inputs'] [size_train + size_val:, ]

target_train = training_data ['targets'] [0:size_train, ]
target_val = training_data ['targets'] [size_train:size_train + size_val, ]
target_test = training_data ['targets'] [size_train + size_val:, ]


##################################################################################################
####  SOME FUNCTIONS and constants      ##########################################################
##################################################################################################


# a list of the activation functions that can be used
act_fun_list = ['linear', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']

# a list of optimizers
optimizer_list = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl']

# a list regression losses
losses_list = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
               'mean_squared_logarithmic_error', 'cosine_similarity', 'logcosh']


# print model results
def model_weights (model):
    model.layers [0].get_weights ()
    weights = model.layers [0].get_weights () [0]
    bias = model.layers [0].get_weights () [1]
    st.write ('#### Results: weights and bias ')
    st.write ('weights --> ', str (weights))
    # st.write ('a --> ' + str (weights [0]) + '; b --> ' + str (weights [1]) + '; c --> ' + str (weights [2]))
    st.write ('bias --> ', str (bias))


# function to plot loss values vs validation loss values
def plot_ls_val (history):
    history_df = pd.DataFrame (history.history)
    fig = go.Figure ()
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['loss'],
                               mode='lines',
                               name='loss'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['val_loss'],
                               mode='lines+markers',
                               name='val_loss',
                               line=dict (color="red")))
    st.plotly_chart (fig, use_container_width=True)


# function to plot different loss functions selected in the compile, metrics params
def plot_losses (history):
    history_df = pd.DataFrame (history.history)
    fig = go.Figure ()
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['loss'],
                               mode='lines',
                               name='loss-selected'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['mse'],
                               mode='lines',
                               line=dict (color="brown"),
                               name='mean_squared_error'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['mae'],
                               mode='lines',
                               line=dict (color="green"),
                               name='mean_absolute_errors'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['cosine_proximity'],
                               mode='lines',
                               line=dict (color="pink"),
                               name='cosine_proximity'))

    st.plotly_chart (fig, use_container_width=True)


# Function to plot real vs predicted values by the model
def plot_real_pred (model, targettest):
    target_test = np.squeeze (targettest)  # to remove square brackts of the numpy nd array
    data_pred = {'Predict': model.predict (data_test).flatten (),
                 'Real': target_test}
    predict_df = pd.DataFrame (data_pred)

    fig2 = go.Figure ()
    fig2.add_trace (go.Scatter (x=predict_df.index, y=predict_df ['Real'],
                                mode='lines',
                                name='Real data'))
    fig2.add_trace (go.Scatter (x=predict_df.index, y=predict_df ['Predict'],
                                mode='markers',
                                name='Prediction',
                                line=dict (color="red")))
    st.plotly_chart (fig2, use_container_width=True)


# Function to print the evaluated values by the model

def print_evaluate(model):
    evaluate_A = model.evaluate (data_test, target_test, batch_size=128)
    st.markdown (f"""
    * **Loss by selected function:** {round(evaluate_A[0],4)}
    * **Loss by mean_squared_error:**  {round(evaluate_A [1],4)}
    * **Loss by mean_absolute_errors:**  {round(evaluate_A [2],4)}
    * **Loss by cosine proximity:** {round(evaluate_A [3],4)}
    """)



##################################################################################################
####  SPLIT SCREEN TO COMPARE MODELS       #######################################################
##################################################################################################

col3, col4 = st.columns ([1, 1])


##################################################################################################
# LEFT SCREEN
##################################################################################################

with col3:
    # model epochs
    st.write ('#### Select number of epochs - A ')
    num_epochs_A = st.slider ('### epochs A', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write ('#### Select the activation function - A ')
    activation_f_A = st.radio ("### Choose activation function A", act_fun_list)

    model_A = tf.keras.Sequential ([tf.keras.layers.Dense (
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
        bias_constraint=None, )])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write ('#### Select an optimizer - A ')
    optimizer_sel = st.radio ("### Choose an optimizer A", optimizer_list)
    learn_rate_A = st.slider ('Learning rate of optimizers A: ', min_value=0.0001, max_value=0.5000, step=0.0001,
                              value=0.02, format="%f")

    optimizer_A = f"tf.keras.optimizers.{optimizer_sel}(learning_rate={learn_rate_A})"
    optimizer_A = eval (optimizer_A)

    st.write ('#### Select regression losses  - A ')
    losses_A = st.radio ("### Choose regression losses A", losses_list)

    model_A.compile (optimizer=optimizer_A,
                     loss=losses_A,
                     metrics=['mse', 'mae', 'cosine_proximity']
                     )

    # train the model
    history_A = History ()
    model_A.fit (data_train, target_train,
                 epochs=num_epochs_A,
                 verbose=2,
                 validation_data=(data_val, target_val),
                 callbacks=[history_A])

    # Print model results
    model_weights (model_A)

    # Graph with loss and validation set loss
    st.write ('#### Graphs of loss vs validation loss ')
    plot_ls_val (history_A)

    # Graph of different losses
    st.write ('#### Graphs of different losses ')
    plot_losses (history_A)

    # Evaluating the model
    st.write ('#### Evaluation A  ')
    print_evaluate(model_A)


    # Predict vs actual values
    st.write ('#### Graphs of Real vx Predicted case A ')
    plot_real_pred (model_A, target_test)


##################################################################################################
# RIGHT SCREEN
##################################################################################################
with col4:
    # model epochs
    st.write ('#### Select number of epochs - B ')
    num_epochs_B = st.slider ('### epochs B', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write ('#### Select the activation function - B ')
    activation_f_B = st.radio ("### Choose activation function B", act_fun_list)
    model_B = tf.keras.Sequential ([tf.keras.layers.Dense (
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
        bias_constraint=None, )])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write ('#### Select an optimizer - B ')
    optimizer_sel_B = st.radio ("### Choose an optimizer B", optimizer_list)
    learn_rate_B = st.slider ('Learning rate of optimizers B: ', min_value=0.0001, max_value=0.5000, step=0.0001,
                              value=0.02, format="%f")

    optimizer_B = f"tf.keras.optimizers.{optimizer_sel_B}(learning_rate={learn_rate_B})"
    optimizer_B = eval (optimizer_B)

    st.write ('#### Select regression losses  - B ')
    losses_B = st.radio ("### Choose regression losses B", losses_list)

    model_B.compile (optimizer=optimizer_B,
                     loss=losses_B,
                     metrics=['mse', 'mae', 'cosine_proximity']
                     )

    # train the model

    history_B = History ()
    model_B.fit (data_train, target_train,
                 epochs=num_epochs_B,
                 verbose=0,
                 validation_data=(data_val, target_val),
                 callbacks=[history_B])  # verbose 2 para ver todo

    # Print model results
    model_weights (model_B)

    # Graph with loss and validation set loss
    st.write ('#### Graphs of loss vs validation loss ')
    plot_ls_val (history_B)

    # Graph of different losses
    st.write ('#### Graphs of different losses ')
    plot_losses (history_B)

    # Evaluating the model
    st.write ('#### Evaluation B  ')
    print_evaluate (model_B)

    # Predict vs actual values
    st.write ('#### Graphs of Real vx Predicted case A ')
    plot_real_pred (model_B, target_test)
