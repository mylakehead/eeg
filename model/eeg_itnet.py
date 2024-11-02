import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Dropout, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Concatenate, Dense, Flatten, AveragePooling2D
from tensorflow.keras.constraints import MaxNorm


class EEGItNetModel:
    # Inception Temporal Convolutional Network
    def __init__(self, chans, samples):
        self.chans = chans
        self.samples = samples
        self.frequency_filters = [2, 4, 8]
        self.spatial_filters = [1, 1, 1]

    def build(self):
        input_block = Input(shape=(self.chans, self.samples, 1))

        block_1 = Conv2D(
            self.frequency_filters[0], (1, 16), use_bias=False, activation='linear', padding='same'
        )(input_block)
        block_1 = BatchNormalization()(block_1)
        block_1 = DepthwiseConv2D(
            (self.chans, 1), use_bias=False, padding='valid', depth_multiplier=self.spatial_filters[0],
            activation='linear', depthwise_constraint=MaxNorm(max_value=1), name='Spatial_filter_1'
        )(block_1)
        block_1 = BatchNormalization()(block_1)
        # TODO
        _ = Activation('elu')(block_1)

        return


def inception_temporal_convolutional_net(chans, samples):
    drop_rate = 0.4
    n_ff = [2, 4, 8]
    n_sf = [1, 1, 1]

    input_block = Input(shape=(chans, samples, 1))

    block1 = Conv2D(n_ff[0], (1, 16), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_1')(input_block)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[0], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_1')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    block2 = Conv2D(n_ff[1], (1, 32), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_2')(input_block)
    block2 = BatchNormalization()(block2)
    block2 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[1], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_2')(block2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    block3 = Conv2D(n_ff[2], (1, 64), use_bias=False, activation='linear', padding='same',
                    name='Spectral_filter_3')(input_block)
    block3 = BatchNormalization()(block3)
    block3 = DepthwiseConv2D((chans, 1), use_bias=False, padding='valid', depth_multiplier=n_sf[2], activation='linear',
                             depthwise_constraint=MaxNorm(max_value=1),
                             name='Spatial_filter_3')(block3)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)

    block = Concatenate(axis=-1)([block1, block2, block3])

    block = AveragePooling2D((1, 4))(block)
    block_in = Dropout(drop_rate)(block)

    paddings = [[0, 0], [0, 0], [3, 0], [0, 0]]
    block = tf.keras.ops.pad(block_in, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_in, block])

    paddings = [[0, 0], [0, 0], [6, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    paddings = [[0, 0], [0, 0], [12, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    paddings = [[0, 0], [0, 0], [24, 0], [0, 0]]
    block = tf.keras.ops.pad(block_out, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block = tf.keras.ops.pad(block, paddings, "constant")
    block = DepthwiseConv2D((1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)
    block = Dropout(drop_rate)(block)
    block_out = Add()([block_out, block])

    block = block_out

    block = Conv2D(28, (1, 1))(block)
    block = BatchNormalization()(block)
    block = Activation('elu')(block)

    block = AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(block)

    block = Dropout(drop_rate)(block)
    embedded = Flatten()(block)

    out_class = 4
    out = Dense(out_class, activation='softmax', kernel_constraint=MaxNorm(0.25))(embedded)

    model = Model(inputs=input_block, outputs=out)
    return model
