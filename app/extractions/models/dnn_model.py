import tensorflow as tf

from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model

# from tensorflow_addons.activations import tanhshrink
# tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink})

from extractions.models.cross_section_layer import TotalFLayer
from extractions.models.split_layer import SplitLayer

from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LEARNING_RATE
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5

def full_cross_section_dnn_model():
    
    try:

        # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
        initializer = tf.keras.initializers.RandomUniform(
            minval = -0.1,
            maxval = 0.1,
            seed = None)

        # (2) Make the TF Input Layer:
        input_kinematics = Input(shape = (5, ), name = 'input_layer')
        # QQ, x_b, t, phi, k = tf.split(input_kinematics, num_or_size_splits = 5, axis = 1)

        QQ = Lambda(lambda x: x[:, 0:1])(input_kinematics)
        x_b = Lambda(lambda x: x[:, 1:2])(input_kinematics)
        t = Lambda(lambda x: x[:, 2:3])(input_kinematics)
        phi = Lambda(lambda x: x[:, 3:4])(input_kinematics)
        k = Lambda(lambda x: x[:, 4:5])(input_kinematics)

        input_kinematics_subnet = Concatenate(axis=1)([QQ, x_b, t])
    
        # (3): Define the Model Architecture:
        x1 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1, activation = "relu", kernel_initializer = initializer)(input_kinematics_subnet)
        x2 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2, activation = "tanh", kernel_initializer = initializer)(x1)
        x3 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3, activation = "relu", kernel_initializer = initializer)(x2)
        x4 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4, activation = "relu", kernel_initializer = initializer)(x3)
        output_cffs = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5, activation = "linear", kernel_initializer = initializer, name = 'cff_output_layer')(x4)

        # # (4): Combine the kinematics as a single list:
        total_FInputs = Concatenate(axis = 1)([input_kinematics, output_cffs])

        # # (8): Compute, algorithmically, the cross section:
        TotalF = TotalFLayer()(total_FInputs)

        # # (9): Define the model as as Keras Model:
        tfCrossSectionModel = Model(inputs = input_kinematics, outputs = TotalF, name = "cross-section-model")

        print(tfCrossSectionModel.summary())

        # (9): Compile the model with a fixed learning rate using Adam and a Loss of MSE:
        tfCrossSectionModel.compile(
            optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
            loss = tf.keras.losses.MeanSquaredError()
        )   

        # (10): Return the model:
        return tfCrossSectionModel

    except Exception as ERROR:
        print(f"> Error in running DNN model:\n{ERROR}")

        return 0.