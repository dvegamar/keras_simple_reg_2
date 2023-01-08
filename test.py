import numpy as np


training_data = np.load('data_all.npz')

size = len(training_data['inputs'])
size_train = int(0.8 * size)
size_val = int(0.1 * size)
size_test = int(0.1 * size)

data_train= training_data['inputs'][0:size_train,]
data_val= training_data['inputs'][size_train:size_train+size_val,]
data_test= training_data['inputs'][size_train+size_val:-1,]

target_train= training_data['targets'][0:size_train,]
target_val= training_data['targets'][size_train:size_train+size_val,]
target_test= training_data['targets'][size_train+size_val:,]

target_test = np.squeeze(target_test)

print (type(target_test))
print(target_test)





history_df = pd.DataFrame (history_A.history)
    fig = go.Figure ()

    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['loss'],
                               mode='lines',
                               name='loss'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['val_loss'],
                               mode='lines+markers',
                               name='val_loss',
                               line=dict (color="red")))
    st.plotly_chart (fig, use_container_width=True)

