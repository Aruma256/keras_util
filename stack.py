def stack(input, *nodes):
    x = input
    for node in nodes:
        if callable(node):
            x = node(x)
        elif isinstance(node, list):
            x = [stack(x, branch) for branch in node]
        elif isinstance(node, tuple):
            x = stack(x, *node)
        else:
            x = node
    return x


'''
Examples
 example_0 : sequential
 example_1 : 2 forks and merges
 example_2 : more forks and merges
'''


from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, Embedding, LSTM
from keras.layers import Add, Multiply, Concatenate
from keras.utils import plot_model

def example_0():
    input = Input((10,))
    output = stack(
        input,
        Dense(10, activation='relu'),
        Dense(10, activation='relu')
        )
    model = Model(input, output)
    plot_model(model, to_file='example_0.png', show_shapes=True, show_layer_names=False)

def example_1():
    input = Input((10,))
    output = stack(
        input,
        Dense(10),
        [Dense(11, activation='relu'),
         Dense(11, activation='relu')],
        Add(),
        [Dense(12, activation='relu'),
         Dense(12, activation='relu')],
        Multiply(),
        )
    model = Model(input, output)
    plot_model(model, to_file='example_1.png', show_shapes=True, show_layer_names=False)

def example_2():
    input = Input((10,))
    outputs = stack(
        input,
        Dense(11),
        Activation('relu'),
        [(Dense(12),
          Activation('relu'),
          Dense(16)),
         (Dense(13),
          [Dense(14, activation='relu'),
           Dense(14, activation='relu')],
          Add(),
          Dense(16))],
        Add(),
        Activation('relu'),
        [Dense(16, activation='relu'),
         Dense(17, activation='relu')]
        )
    model = Model(input, outputs)
    plot_model(model, to_file='example_2.png', show_shapes=True, show_layer_names=False)

def example_residual_connection():
    input = Input(shape=(256, 256, 3))
    output = stack(
        input,
        [(Conv2D(3, (3, 3), padding='same'),
          Activation('relu'),
          Conv2D(3, (3, 3), padding='same')),
         lambda x: x],
        Add(),
        Activation('relu')
    )
    model = Model(input, output)
    plot_model(model, to_file='example_residual_connection.png', show_shapes=True, show_layer_names=False)

def example_multi_input_and_multi_output():
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    auxiliary_input = Input(shape=(5,), name='aux_input')
    outputs = stack(
        main_input,
        Embedding(output_dim=512, input_dim=10000, input_length=100),
        LSTM(32),
        [Dense(1, activation='sigmoid', name='aux_output'),
         ([auxiliary_input, lambda x: x],
          Concatenate(),
          Dense(64, activation='relu'),
          Dense(64, activation='relu'),
          Dense(64, activation='relu'),
          Dense(1, activation='sigmoid', name='main_output')
         )
        ]
    )
    model = Model([main_input, auxiliary_input], outputs)
    plot_model(model, to_file='example_multi_input_and_multi_output.png')
          
