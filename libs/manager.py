

class MaNNager:
    """
    Neural Network Manager:
        - Holds 
            - [tf.model] a layout of a neural network
            - [function(input dataframe) -> x_input, y_input] 
                an instruction that describes how to 
                create an input layer data out of a dataframe
            - [string] a description of the managed neural network
        - Able to
            - create, train, test a neural network
            - save, load its state
    """

    def __init__(self, model=None, instruction=lambda df: (df.iloc[:, :-1], df.iloc[:, -1]),
                 description='No description provided', **kwargs):
        """
            Parameters:
                - model: tf.keras.Model, a neural network model
                - instruction: function(dataframe) -> x_input, y_input, a function that converts
                    a dataframe to an input layer
                - description: string, a description of the neural network
                - kwargs: additional arguments; self acquires them as attributes
        """
        self.model = model
        self.instruction = instruction
        self.description = description
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    @staticmethod
    def dataframe_split(data, split_size):
        """
            Randomly splits data at given ratio

            Parameters:
                - data: pandas.DataFrame or equivalent, data to split
                - split_size: float, split ratio

            Returns:
                - train_data: pandas.DataFrame or equivalent, (1 - split_size) of data
                - test_data: pandas.DataFrame or equivalent, split_size of data
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(data, test_size=split_size)

    def train(self, train_data, epochs, batch_size, validation_split=None):
        """
            Trains the neural network

            Parameters:
                - train_data: pandas.DataFrame or equivalent, features + labels
                - epochs: int, number of epochs
                - batch_size: int, batch size
                - validation_split: float, split ratio for validation data. If not provided
                    , no validation is performed

            Returns: a dictionary with following keys:
                - history: tf.keras.callbacks.History, training history
                - train_data: pandas.DataFrame or equivalent, dataframe used for training
                - val_data: pandas.DataFrame or equivalent, dataframe used for validation. Present
                    only if validation_split is provided
        """
        if validation_split is None:
            history = self.model.fit(
                *self.instruction(train_data), epochs=epochs, batch_size=batch_size)
            return {'history': history, 'train_data': train_data}
        else:
            actual_train_data, val_data = MaNNager.dataframe_split(
                train_data, validation_split)
            history = self.model.fit(*self.instruction(actual_train_data), epochs=epochs,
                                     batch_size=batch_size, validation_data=self.instruction(val_data))
            return {'history': history, 'train_data': actual_train_data,
                    'val_data': val_data}

    def test(self, test_data, batch_size):
        """
            Tests the neural network

            Parameters:
                - test_data: pandas.DataFrame or equivalent, features + labels
                - batch_size: int, batch size

            Returns: a list of
                - loss: float, loss on test data
                - accuracy: float, accuracy on test data
        """
        return self.model.evaluate(*self.instruction(test_data), batch_size=batch_size)

    def data_to_input_layer(self, data):
        """
            Converts a dataframe to an input layer. Synonym for self.instruction(data)
            
            Parameters:
                - data: pandas.DataFrame or equivalent, input data
                
            Returns:
                - x_input: pandas.DataFrame or equivalent, input layer
                - y_input: pandas.DataFrame or equivalent, output layer
        """
        return self.instruction(data)

    def save(self, path):
        """
            Saves the current manager to a file

            Parameters:
                - path: string, path to a file
        """
        import dill
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @staticmethod
    def load(path):
        """
            Loads a manager from a file

            Parameters:
                - path: string, path to a file

            Returns:
                - a manager
        """
        import dill
        with open(path, 'rb') as f:
            return dill.load(f)

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description
