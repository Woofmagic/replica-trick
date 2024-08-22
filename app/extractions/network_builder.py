class NetworkBuilder:
    def __init__(self, verbose = True):

        # DEBUGGING:
        self.verbose = verbose

        # CONSOLE/TERMINAL MESSAGES:

        # INPUTS:
        self._INPUT_PROMPT_NUMBER_1_NUMBER_OF_ANN_LAYERS = "> How many layers do you want in your ANN architecture?"
        self._INPUT_PROMPT_NUMBER_2_NODES_PER_LAYER = "> Choose how many nodes you want for layer {}."
        self._INPUT_PROMPT_NUMBER_3_ACTIVATION_FUNCTION = "> Choose an activation function for layer {}."
        self._INPUT_PROMPT_NUMBER_4_ACTIVATION_FUNCTION = "> Please select the network's loss function."
        self._INPUT_PROMPT_NUMBER_5_INPUT_SHAPE = "> Please specify the number of inputs of your network."

        # ERRORS
        self._ERROR_PROMPT_NUMBER_1_LAYER_NUMBER_NOT_INT = "> Layer number must be a positive, nonzero integer. Continuing..."
        self._ERROR_PROMPT_NUMBER_2_NODES_PER_LAYER_NOT_INTEGER = "> Number of nodes per layer must be an integer type. Continuing..."
        self._ERROR_PROMPT_NUMBER_3_ACTIVATION_FUNCTION_NOT_STRING = "> Activation functions are of string datatype. Continuing..."
        self._ERROR_PROMPT_NUMBER_3_ACTIVATION_FUNCTION_NOT_IN_LIST = "> User input of `{}` is not included in known list of activation functions."
        self._ERROR_PROMPT_NUMBER_4_LOSS_FUNCTION_NOT_STRING = "> The provided loss function was not a string type. Provide one that is. Continuing..."
        self._ERROR_PROMPT_NUMBER_4_LOSS_FUNCTION_NOT_IN_LIST = "> User input of `{}` is not included in known list of loss functions."
        self._ERROR_PROMPT_NUMBER_5_INPUT_SHAPE_INTEGER = "> The user input of {} wasn't an integer."
        self._ERROR_PROMPT_NUMBER_5_INPUT_SHAPE_NEGATIVE = "> Obviously, the size of the input vector cannot be zero or negative."

        # ACTIVATION FUNCTION NAMES:
        self._ACTIVATION_FUNCTIONS_STRING_ELU = "elu"
        self._ACTIVATION_FUNCTIONS_STRING_EXPONENTIAL = "exponential"
        self._ACTIVATION_FUNCTIONS_STRING_GELU = "gelu"
        self._ACTIVATION_FUNCTIONS_STRING_HARD_SIGMOID = "hard_sigmoid"
        self._ACTIVATION_FUNCTIONS_STRING_LINEAR = "linear"
        self._ACTIVATION_FUNCTIONS_STRING_MISH = "mish"
        self._ACTIVATION_FUNCTIONS_STRING_RELU = "relu"
        self._ACTIVATION_FUNCTIONS_STRING_SELU = "selu"
        self._ACTIVATION_FUNCTIONS_STRING_SIGMOID = "sigmoid"
        self._ACTIVATION_FUNCTIONS_STRING_SOFTMAX = "softmax"
        self._ACTIVATION_FUNCTIONS_STRING_SOFTPLUS = "softplus"
        self._ACTIVATION_FUNCTIONS_STRING_SOFTSIGN = "softsign"
        self._ACTIVATION_FUNCTIONS_STRING_SWISH = "swish"
        self._ACTIVATION_FUNCTIONS_STRING_TANH = "tanh"


        # ACTIVATION FUNCTION ARRAY:
        self._ARRAY_OF_ACCEPTABLE_ACTIVATION_FUNCTIONS = [
            self._ACTIVATION_FUNCTIONS_STRING_ELU,
            self._ACTIVATION_FUNCTIONS_STRING_EXPONENTIAL,
            self._ACTIVATION_FUNCTIONS_STRING_GELU,
            self._ACTIVATION_FUNCTIONS_STRING_HARD_SIGMOID,
            self._ACTIVATION_FUNCTIONS_STRING_LINEAR,
            self._ACTIVATION_FUNCTIONS_STRING_MISH,
            self._ACTIVATION_FUNCTIONS_STRING_RELU,
            self._ACTIVATION_FUNCTIONS_STRING_SELU,
            self._ACTIVATION_FUNCTIONS_STRING_SIGMOID,
            self._ACTIVATION_FUNCTIONS_STRING_SOFTMAX,
            self._ACTIVATION_FUNCTIONS_STRING_SOFTPLUS,
            self._ACTIVATION_FUNCTIONS_STRING_SOFTSIGN,
            self._ACTIVATION_FUNCTIONS_STRING_SWISH,
            self._ACTIVATION_FUNCTIONS_STRING_TANH,
        ]

        # LOSS FUNCTION NAMES:
        self._LOSSES_STRING_BINARY_CROSSENTROPY = "binary crossentropy"
        self._LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY = "binary focal crossentropy"
        self._LOSSES_STRING_CATEGORICAL_CROSSENTROPY = "categorical crossentropy"
        self._LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY = "categorical focal crossentropy"
        self._LOSSES_STRING_CATEGORICAL_HINGE = "cateogrical hinge"
        self._LOSSES_STRING_COSINE_SIMILARITY = "cosine similarity"
        self._LOSSES_STRING_HINGE = "hinge"
        self._LOSSES_STRING_HUBER = "huber"
        self._LOSSES_STRING_KL_DIVERGENCE = "kl divergence"
        self._LOSSES_STRING_LOG_COSH = "log cosh"
        self._LOSSES_STRING_LOSS = "loss"
        self._LOSSES_STRING_MEAN_ABSOLUTE_ERROR = "mean absolute error"
        self._LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR = "mean absolute percentage error"
        self._LOSSES_STRING_MEAN_SQUARED_ERROR = "mean squared error"
        self._LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR = "mean squared logarithmic error"
        self._LOSSES_STRING_POISSON = "poisson"
        self._LOSSES_STRING_REDUCTION = "reduction"
        self._LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY = "sparce categorical crossentropy"
        self._LOSSES_STRING_SQUARED_HINGE = "squared hinge"

        # LOSS FUNCTION ARRAY:
        self._ARRAY_OF_ACCEPTABLE_LOSS_FUNCTIONS = [
            self._LOSSES_STRING_BINARY_CROSSENTROPY,
            self._LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY,
            self._LOSSES_STRING_CATEGORICAL_CROSSENTROPY,
            self._LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY,
            self._LOSSES_STRING_COSINE_SIMILARITY,
            self._LOSSES_STRING_CATEGORICAL_HINGE,
            self._LOSSES_STRING_HUBER,
            self._LOSSES_STRING_KL_DIVERGENCE,
            self._LOSSES_STRING_LOG_COSH,
            self._LOSSES_STRING_LOSS,
            self._LOSSES_STRING_MEAN_ABSOLUTE_ERROR,
            self._LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR,
            self._LOSSES_STRING_MEAN_SQUARED_ERROR,
            self._LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR,
            self._LOSSES_STRING_POISSON,
            self._LOSSES_STRING_REDUCTION,
            self._LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY,
            self._LOSSES_STRING_SQUARED_HINGE
        ]

        # LOSS FUNCTION DICTIONARY:
        self._DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS = {
            self._LOSSES_STRING_BINARY_CROSSENTROPY: tensorflow.keras.losses.BinaryCrossentropy(),
            self._LOSSES_STRING_BINARY_FOCAL_CROSSENTROPY: tensorflow.keras.losses.BinaryFocalCrossentropy(),
            self._LOSSES_STRING_CATEGORICAL_CROSSENTROPY: tensorflow.keras.losses.CategoricalCrossentropy(),
            self._LOSSES_STRING_CATEGORICAL_FOCAL_CROSSENTROPY: tensorflow.keras.losses.CategoricalFocalCrossentropy(),
            self._LOSSES_STRING_CATEGORICAL_HINGE: tensorflow.keras.losses.CategoricalHinge(),
            self._LOSSES_STRING_COSINE_SIMILARITY: tensorflow.keras.losses.CosineSimilarity(),
            self._LOSSES_STRING_HINGE: tensorflow.keras.losses.Hinge(),
            self._LOSSES_STRING_HUBER: tensorflow.keras.losses.Huber(),
            self._LOSSES_STRING_KL_DIVERGENCE: tensorflow.keras.losses.KLDivergence(),
            self._LOSSES_STRING_LOG_COSH: tensorflow.keras.losses.LogCosh(),
            self._LOSSES_STRING_LOSS: tensorflow.keras.losses.Loss(),
            self._LOSSES_STRING_MEAN_ABSOLUTE_ERROR: tensorflow.keras.losses.MeanAbsoluteError(),
            self._LOSSES_STRING_MEAN_ABSOLULTE_PERCENTAGE_ERROR: tensorflow.keras.losses.MeanAbsolutePercentageError(),
            self._LOSSES_STRING_MEAN_SQUARED_ERROR: tensorflow.keras.losses.MeanSquaredError(),
            self._LOSSES_STRING_MEAN_SQUARED_LOGARITHMIC_ERROR: tensorflow.keras.losses.MeanSquaredLogarithmicError(),
            self._LOSSES_STRING_POISSON: tensorflow.keras.losses.Poisson(),
            self._LOSSES_STRING_REDUCTION: tensorflow.keras.losses.Reduction(),
            self._LOSSES_STRING_SPARCE_CATEGORICAL_CROSSENTROPY: tensorflow.keras.losses.SparseCategoricalCrossentropy(),
            self._LOSSES_STRING_SQUARED_HINGE: tensorflow.keras.losses.SquaredHinge(),
            }
        
        # ===== MODEL NAME =====
        self.model_name = None

        # ===== MODEL METAPARAMETERS =====

        # (1): Number of Hidden Layers:
        self.number_of_hidden_layers = None

        # (2): A List of the Number of Nodes per Layer:
        self.list_of_number_of_nodes_per_layer = None

        # (3): A List of Activation Functions per Layer:
        self.list_of_activation_functions_for_each_layer = []

        # (4): Keras Loss Function:
        self.model_loss_function = None

        # (5): Model Input Dimension
        self.number_of_input_variables = 0
    
        # (6): Model Input Layer:
        self.model_input_layer = None

        # (7): Model Output:
        self.model_output_layer = None

    # ===== CLASS METHODS =====

    def obtain_number_of_ANN_hidden_layers(self):
        """
        Description
        --------------
        Obtain the number of hidden layers for the ANN from the user.

        
        Parameters
        --------------
        verbose (boolean):
            Do you want to see all output of this function evaluation?
            
        Notes
        --------------
        (1) We just need to obtain a nonzero, positive integer that
        represents the number of layers in the ANN.

        (2) https://stackoverflow.com/a/23294659 -> For a healthy way
        to construct a while loop like this.
        """
        while True:

            try:
                
                number_of_ANN_layers = int(input(self._INPUT_PROMPT_NUMBER_1_NUMBER_OF_ANN_LAYERS))
                
                if self.verbose:
                    print(f"> Received input: {number_of_ANN_layers} ({type(number_of_ANN_layers)}).")

            except ValueError:

                print(self._ERROR_PROMPT_NUMBER_1_LAYER_NUMBER_NOT_INT)
                continue

            if number_of_ANN_layers <= 0:

                print(self._ERROR_PROMPT_NUMBER_1_LAYER_NUMBER_NOT_INT)
                continue

            else:
                
                print(f"> Received input: {number_of_ANN_layers} ({type(number_of_ANN_layers)}). Exiting...")
                break

        if self.verbose:
            print(f"> User defined {number_of_ANN_layers} Layers in the network.")

        self.number_of_hidden_layers = number_of_ANN_layers
        return None
    
    def obtain_nodes_per_layer(self, number_of_ANN_layers):
        """
        Description
        --------------
        Obtain the number of nodes per layer in the ANN.

        
        Parameters
        --------------
        number_of_ANN_layers (int):
            the number of layers in the ANN

        verbose (boolean):
            prints the intermediate computations of the function
        
        Notes
        --------------
        (1) For all each layer, we need to populate it with a number of neurons.
            So, this function is about obtaining a list of intergers that correspond
            to the number of neurons per layer.

        (2) The output is a list of integers.
        """
        list_of_number_of_nodes_per_layer = []

        for layer_index in range(number_of_ANN_layers):

            while True:

                try:

                    number_of_nodes_per_given_layer = int(input(self._INPUT_PROMPT_NUMBER_2_NODES_PER_LAYER.format(layer_index + 1)))

                    if self.verbose:
                        print(f"> Received input: {number_of_nodes_per_given_layer} ({type(number_of_nodes_per_given_layer)}).")

                except ValueError:

                    print(self._ERROR_PROMPT_NUMBER_2_NODES_PER_LAYER_NOT_INTEGER)
                    continue

                print(f"> Received input: {number_of_nodes_per_given_layer} ({type(number_of_nodes_per_given_layer)}). Exiting...")
                list_of_number_of_nodes_per_layer.append(number_of_nodes_per_given_layer)
                break
            
            print(f"> User defined Layer #{layer_index + 1} to have {number_of_nodes_per_given_layer} nodes.")

        self.list_of_number_of_nodes_per_layer = list_of_number_of_nodes_per_layer
        return None
    
    def check_if_activation_function_included(self, user_entered_activation_function = None):
        """
        Description
        --------------
        Sanitize/verify that the user-typed string corresponds to a given and known
        activation function. Checks against the TensorFlow nomenclature.

        
        Parameters
        --------------
        user_entered_activation_function (string):
            the name of the desired activation function for the layer of nodes
            
        
        Notes
        --------------
        (1): At this stage, we are actually validating the list of valid activation function
            names. There's no real reason to do this, but it keeps us accountable.

        """

        if user_entered_activation_function == None:
            return False
        else:
            if user_entered_activation_function in self._ARRAY_OF_ACCEPTABLE_ACTIVATION_FUNCTIONS:
                return True
            else:
                return False
            
    def construct_array_of_layer_activation_functions(self, number_of_ANN_layers):
        """
        Description
        --------------
        Obtain the activation functions for each neuron in a given layer 
        from the user. There are only a few acceptable options for the 
        types of activation functions.


        Parameters
        --------------
        number_of_ANN_layers (int): 
            The number of layers, not including the input data, in the network.
        
            
        Notes
        --------------
        (1) For all neurons in a given layer, we will apply the same activation function.
        """
        list_of_activation_functions_for_each_layer = []

        for layer_index in range(number_of_ANN_layers):

            while True:

                try:

                    user_proposed_activation_function = str(input(self._INPUT_PROMPT_NUMBER_3_ACTIVATION_FUNCTION.format(layer_index + 1)))
                    if self.verbose:
                        print(f"> Received input: {user_proposed_activation_function} ({type(user_proposed_activation_function)}).")

                except ValueError:

                    print(self._ERROR_PROMPT_NUMBER_3_ACTIVATION_FUNCTION_NOT_STRING)
                    continue

                if not self.check_if_activation_function_included(user_proposed_activation_function):

                    print(self._ERROR_PROMPT_NUMBER_3_ACTIVATION_FUNCTION_NOT_IN_LIST.format(user_proposed_activation_function))
                    continue

                else:

                    print(f"> Received input: {user_proposed_activation_function} ({type(user_proposed_activation_function)}). Exiting...")
                    list_of_activation_functions_for_each_layer.append(user_proposed_activation_function)
                    break
            
            print(f"> User defined layer number {layer_index + 1} to use {user_proposed_activation_function} activation.")

        self.list_of_activation_functions_for_each_layer = list_of_activation_functions_for_each_layer
        return None

    def check_if_loss_function_included(self, user_entered_loss_function = None):
        """
        Description
        --------------
        Sanitize/verify that the user-typed string corresponds to a given and known
        activation function. Checks against the TensorFlow nomenclature.

        
        Parameters
        --------------
        user_entered_activation_function (string): 
            the name of the desired loss functions
            
        
        Notes
        --------------
        (1): This is just an intermediate validation. We want to make sure we
            are referring to things that exist. That's why we put this 
            intermediate validation here.

        """

        if user_entered_loss_function == None:
            return False
        else:
            if user_entered_loss_function in self._ARRAY_OF_ACCEPTABLE_LOSS_FUNCTIONS:
                return True
            else:
                return False

    def choose_network_loss_function(self):
        """
        Description
        --------------
        We need a loss function in comparing the model's terrible output with
        actual data. All this function involves is choosing the loss function.


        Parameters
        --------------
            verbose (boolean):
                prints out the intermediate steps in a calculation
        
            
        Notes
        --------------
        """
        while True:

            try:

                user_proposed_loss_function = str(input(self._INPUT_PROMPT_NUMBER_4_ACTIVATION_FUNCTION))
                if self.verbose:
                    print(f"> Received user input of {user_proposed_loss_function} ({type(user_proposed_loss_function)}).")

            except ValueError:

                print(self._ERROR_PROMPT_NUMBER_4_LOSS_FUNCTION_NOT_STRING)
                continue

            if not self.check_if_loss_function_included(user_proposed_loss_function):

                print(self._ERROR_PROMPT_NUMBER_4_LOSS_FUNCTION_NOT_IN_LIST.format(user_proposed_loss_function))
                continue

            else:

                print(f"> Received user input of {user_proposed_loss_function} ({type(user_proposed_loss_function)}). Exiting...")
                break
            
        self.user_entered_loss_function = user_proposed_loss_function.lower()
        self.model_loss_function = self.translate_user_input_loss_function_to_keras_loss_function(user_proposed_loss_function)
        return None
    
    def translate_user_input_loss_function_to_keras_loss_function(self, string_user_input_loss_function):
        """
        Description
        --------------
        All this function does is it takes a string of user input that is supposed to
        represent the name of some popular loss function (those included in TensorFlow)
        and attempt to match it with the existing classes available. If the class
        exists, we return that TF instance. If the class does not, then we tell the user
        to get good.

        Parameters
        --------------
        string_user_input_loss_function (string)
            the name of the desired loss function as a string
        
            
        Notes
        --------------
        (1): We reference a huge dictionary.
        """
        try:
            return self._DICTIONARY_MAP_USER_INPUT_TO_KERAS_LOSS.get(string_user_input_loss_function, None)
        
        except Exception as E:
            print(f"> Error in translating input: {string_user_input_loss_function} to Keras Loss class instance:\n> {E}")
            return None

    def obtain_number_of_input_variables(self):
        """
        Description
        --------------
        The number of input variables in the feed-forward network.


        Parameters
        --------------
        number_of_input_variables (int): 
            The number of input variables we want to feed into the network.
        
            
        Notes
        --------------
        (1) Unusually, the input layer is defined according to a "vector" of
            length number_of_input_variables. So, we need to obtain the dimension
            of this "input vector." That's what this function is about.
        """
        while True:

                try:

                    number_of_input_variables = int(input(self._INPUT_PROMPT_NUMBER_5_INPUT_SHAPE))
                    if self.verbose:
                        print(f"> Received input: {number_of_input_variables} ({type(number_of_input_variables)}).")

                except ValueError:

                    print(self._ERROR_PROMPT_NUMBER_5_INPUT_SHAPE_INTEGER.format(number_of_input_variables))
                    continue

                if number_of_input_variables <= 0:

                    print(self._ERROR_PROMPT_NUMBER_5_INPUT_SHAPE_NEGATIVE)
                    continue

                else:

                    print(f"> Received user input of {number_of_input_variables} ({type(number_of_input_variables)}). Exiting...")
                    self.number_of_input_variables = number_of_input_variables
                    break
            
        return None
    
    def calculate_output_layer_as_nested_hidden_layers(self, inputs, number_of_hidden_layers, list_of_nodes_per_layer, list_of_activation_functions):
        """
        Description
        --------------
        This function performs a recursive call to itself in order to properly
        nest every instance of a Keras Dense() instance into an earlier one. That
        is how we can stack Dense layers together.

        Parameters
        --------------
        input (Keras Dense() or Input() instance): 
            the way this works is that we need to stack Dense()() layers
            so that we can get the recursion Dense(N)(Dense(M)(Dense(P)(...))).

        number_of_hidden_layers (int): 
            while we don't use a for-loop below, we are using this integer to 
            effectively perform a loop by decrementing an index number.
        
            
        Notes
        --------------
        None
        """
        if len(list_of_nodes_per_layer) is not len(list_of_activation_functions):
            print(f"> Mismatching length of lists: {len(list_of_nodes_per_layer)} in list of nodes per layer but {len(list_of_activation_functions)} in list of activation functions.")
            return None

        if number_of_hidden_layers == 0:
            return inputs
        
        else:
            
            number_of_nodes_in_this_layer = list_of_nodes_per_layer[number_of_hidden_layers - 1]
            activation_function_in_this_layer = list_of_activation_functions[number_of_hidden_layers - 1]


            if self.verbose:
                print(f"> Now initializing a TF Dense() layer with {number_of_nodes_in_this_layer} nodes and {activation_function_in_this_layer} activations.")
                
                # Kernel Initializers are a little tough. There are two that are decent:
                #
                # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform
                # tensorflow.keras.initializers.RandomUniform minval = -0.1,maxval = 0.1)

                #
                # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
                # tensorflow.keras.initializers.GlorotUniform()
            nested_layer = tensorflow.keras.layers.Dense(
                number_of_nodes_in_this_layer,
                activation = activation_function_in_this_layer,
                use_bias = True,
                kernel_initializer = tensorflow.keras.initializers.RandomUniform(
                    minval = -0.1,
                    maxval = 0.1
                ),
                bias_initializer = 'zeros',
                kernel_regularizer = None,
                bias_regularizer = None,
                activity_regularizer = None,
                kernel_constraint = None,
                bias_constraint = None,
                name = f"hidden_layer_{number_of_hidden_layers}",
                )(self.calculate_output_layer_as_nested_hidden_layers(inputs, number_of_hidden_layers - 1, list_of_nodes_per_layer, list_of_activation_functions))
            return nested_layer
        
    def register_tensorflow_model(self, input_layer, output_layer):
        """
        Description
        --------------
        All this function does is return for us a Keras Model() instance
        with the inputs and outputs kwargs specified. 

        Parameters
        --------------
        input_layer (Keras Input() instance): 
            the actual Keras Input() layer instance

        output_layer (Keras Input() instance): 
            this should be some kind of Keras Layer() instance as 
            obtained by nesting layers together
        
            
        Notes
        --------------
        None
        """

        try:

            new_tensorflow_model = tensorflow.keras.models.Model(
                inputs = input_layer,
                outputs = output_layer,
                name = self.model_name
            )
            self.tensorflow_network = new_tensorflow_model
        
        except Exception as E:
            
            print(f"> Error in constructing a Keras model:\n> {E}")
            return None

    def compile_tensorflow_network(self, tensorflow_network, keras_loss_function):
        """
        Description
        --------------
        Actually compile the TensorFlow ANN with all the details that 
        we supplied earlier. In reality, this function does nothing but
        use the .compile() method, sets the (i) optimizer, (ii) loss,
        and (iii) metrics.

        Parameters
        --------------
        tensorflow_network (int): 
            The number of input variables we want to feed into the network.
        
            
        Notes
        --------------
        (1) Unusually, the input layer is defined according to a "vector" of
            length number_of_input_variables. So, we need to obtain the dimension
            of this "input vector." That's what this function is about.
        """
        if tensorflow_network == None:
            print(f"> No TensorFlow network instance supplied. Exiting...")
            return None
        
        try:
            tensorflow_network.compile(
            optimizer = tensorflow.keras.optimizers.Adam(
                learning_rate = 0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-07,
                amsgrad = False,
                weight_decay = None,
                clipnorm = None,
                clipvalue = None,
                global_clipnorm = None,
                use_ema = False,
                ema_momentum = 0.99,
                ema_overwrite_frequency = None,
                jit_compile = True,
                name='Optimizer_Adam',
            ),
            loss = keras_loss_function,
            metrics = tensorflow.keras.metrics.MeanSquaredError(),
            loss_weights = None,
            weighted_metrics = None,
            run_eagerly = None,
            steps_per_execution = None,
            jit_compile = None,
            pss_evaluation_shards = 0,
        )
            self.tensorflow_network = tensorflow_network
            return None
        except Exception as E:
            print(f"> Error in compiling the TensorFlow network:\n> {E}")
            return None

    def describe_tensorflow_network(self, tensorflow_network):
        """
        Description
        --------------
        TensorFlow's description of the network.

        Parameters
        --------------
        tensorflow_network (int): 
            The instance of the tensorflow network.
        
            
        Notes
        --------------
        None
        """
        try:
            return tensorflow_network.summary()
        
        except Exception as E:
            
            print(f"> Error in describing the TF network:\n> {E}")
            return None

    def construct_model_name(self):
        """
        Description
        --------------
        We build the name of the model:

        Parameters
        --------------
        None    
            
        Notes
        --------------
        None
        """
        number_of_layers = self.number_of_hidden_layers + 1

        if self.verbose:
            print(f"> Total number of layers computed as: {number_of_layers}")

        current_timestamp = self.generate_timestamp()
        
        if current_timestamp == None:
            print(f"> Issue generating model name -- Received NoneType from timestamp.")
            return None
            
        if self.verbose:
            print(f"> Computed current timestamp as: {current_timestamp}")

        model_name = f"model{current_timestamp}_{number_of_layers}{''.join(str(nodes) + word for nodes, word in zip(self.list_of_number_of_nodes_per_layer, self.list_of_activation_functions_for_each_layer))}"
        print(model_name)

        self.model_name = model_name

    def generate_timestamp(self):
        try:

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.verbose:
                print(f"> Computed the timestamp to be: {timestamp}")

            return timestamp
        
        except Exception as E:
            print(f"> Error in generating the timestamp:\n> {E}")
            return None

        
    def build_network(self):
        
        # (1): Obtain the number of layers:
        self.obtain_number_of_ANN_hidden_layers()
        if self.verbose:
            print(f"> [1]: Obtained number of ANN hidden layers.")

        # (2): Obtain the number of nodes per layer:
        self.obtain_nodes_per_layer(self.number_of_hidden_layers)
        if self.verbose:
            print(f"> [2]: Obtained list with number of nodes per layer.")

        # (3): Obtain the activation functions for each layer:
        self.construct_array_of_layer_activation_functions(self.number_of_hidden_layers)
        if self.verbose:
            print(f"> [3]: Obtained activation functions for each layer.")

        # (4): Obtain the network's loss function:
        self.choose_network_loss_function()
        if self.verbose:
            print(f"> [4]: Obtained the network's loss function.")

        # (5): Obtain the Network Input Dimensionality:
        self.obtain_number_of_input_variables()
        if self.verbose:
            print(f"> [5]: Obtained the input layer's dimensionality.")

        # (6): Obtain the Actual Network Inputs (Keras Layers):
        self.model_input_layer = tensorflow.keras.layers.Input(shape = (self.number_of_input_variables, ))
        if self.verbose:
            print(f"> [6]: Registered the network's first layer.")

        # (7): Obtain the Actual Network Outputs (Keras Layers):
        self.model_output_layer = self.calculate_output_layer_as_nested_hidden_layers(
            self.model_input_layer,
            self.number_of_hidden_layers,
            self.list_of_number_of_nodes_per_layer,
            self.list_of_activation_functions_for_each_layer
        )
        if self.verbose:
            print(f"> [7]: Computed the network's output layer.")

        # (8): Construct the name of the Model:
        self.construct_model_name()
        if self.verbose:
            print(f"> [8]: Model name derived.")

        # (9): Register the TensorFlow Model:
        self.register_tensorflow_model(self.model_input_layer, self.model_output_layer)
        if self.verbose:
            print(f"> [9]: Registered TensorFlow model...")

        # (10): Compile the Model:
        self.compile_tensorflow_network(self.tensorflow_network, self.model_loss_function)
        if self.verbose:
            print(f"> [10]: Network compiled!")

        # (11): Describe the Model:
        self.describe_tensorflow_network(self.tensorflow_network)
        if self.verbose:
            print(f"> [11]: Network described.")

        return self.tensorflow_network