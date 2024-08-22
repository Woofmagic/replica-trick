
def build_network():
    # (1): Create a TensorFlow network:
    tensorflow_network_1 = NetworkBuilder().build_network()

    return tensorflow_network_1

def train_network(tf_network, training_x_data, training_y_data):
    # (2): Train the network:
    collect_weights_and_biases_callback = WeightsBiasesCallback(tf_network)
    history_of_training_1 = tf_network.fit(
        training_x_data_rich, 
        training_y_data_rich, 
        epochs = 700,
        # validation_split = 0.2, # https://www.tensorflow.org/guide/keras/training_with_built_in_methods#automatically_setting_apart_a_validation_holdout_set
        callbacks = [collect_weights_and_biases_callback],
        verbose = 1)