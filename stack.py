def stack(input, *nodes):
    x = input
    for node in nodes:
        if isinstance(node, list):
            x = [stack(x, *branch) for branch in node]
        else:
            x = node(x)
    return x


'''
Examples
 example_0 : sequential
 example_1 : 2 forks and merges
 example_2 : more forks and merges
'''


from keras.models import Model
from keras.layers import Input, Dense, Activation, Add, Multiply
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
        [(Dense(11), Activation('relu')),
         (Dense(11), Activation('relu'))],
        Add(),
        [(Dense(12), Activation('relu')),
         (Dense(12), Activation('relu'))],
        Multiply(),
        )
    model = Model(input, output)
    plot_model(model, to_file='example_1.png', show_shapes=True, show_layer_names=False)

def example_2():
    input = Input((10,))
    output = stack(
        input,
        Dense(11),
        Activation('relu'),
        [(Dense(12),
          Activation('relu'),
          Dense(16)),
         (Dense(13),
          [(Dense(14), Activation('relu')),
           (Dense(14), Activation('relu'))],
          Add(),
          Dense(16))],
        Add(),
        Activation('relu'),
        [(Dense(16), Activation('relu')),
         (Dense(17), Activation('relu'))]
        )
    model = Model(input, output)
    plot_model(model, to_file='example_2.png', show_shapes=True, show_layer_names=False)

