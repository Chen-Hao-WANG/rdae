import tensorflow as tf

def create_encoder_rnn_model(input_shape_sequence, # e.g., (seq_len, H, W, channel_num)
                             name="EncoderRNN"):
    """
    Creates an RNN Encoder model using Keras Functional API.
    This model processes an entire sequence.
    CNN operations are applied per time-step using TimeDistributed,
    followed by ConvLSTM2D layers.

    Args:
        input_shape_sequence: Shape of the input sequence (seq_len, H, W, C).
        name: Name for the Keras model.

    Returns:
        A Keras Model representing the RNN encoder.
    """
    
    input_tensor_sequence = tf.keras.Input(shape=input_shape_sequence, name="encoder_input_sequence")
    
    leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)

    td_conv_enc_0 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_0" # padding to lowercase
    )
    td_conv_enc_1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_1" # padding to lowercase
    )
    
    x = td_conv_enc_0(input_tensor_sequence)
    x = td_conv_enc_1(x)
    net1_h_seq = tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_1')(x) # padding to lowercase for ConvLSTM2D
    
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"), name="TD_MaxPool_1")(net1_h_seq) # padding to lowercase
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=43, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_2")(x) # padding to lowercase
    net2_h_seq = tf.keras.layers.ConvLSTM2D(filters=43, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_2')(x) # padding to lowercase

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"), name="TD_MaxPool_2")(net2_h_seq) # padding to lowercase
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=57, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_3")(x) # padding to lowercase
    net3_h_seq = tf.keras.layers.ConvLSTM2D(filters=57, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_3')(x) # padding to lowercase

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"), name="TD_MaxPool_3")(net3_h_seq) # padding to lowercase
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=76, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_4")(x) # padding to lowercase
    net4_h_seq = tf.keras.layers.ConvLSTM2D(filters=76, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_4')(x) # padding to lowercase

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"), name="TD_MaxPool_4")(net4_h_seq) # padding to lowercase
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=101, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_5")(x) # padding to lowercase
    net5_h_seq = tf.keras.layers.ConvLSTM2D(filters=101, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_5')(x) # padding to lowercase

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same"), name="TD_MaxPool_5")(net5_h_seq) # padding to lowercase
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=101, kernel_size=3, strides=1, padding="same", activation=leaky_relu), name="TD_Enc_conv_6")(x) # padding to lowercase
    net6_h_seq = tf.keras.layers.ConvLSTM2D(filters=101, kernel_size=3, padding='same', return_sequences=True, activation=leaky_relu, recurrent_activation='hard_sigmoid', name='ConvLSTM_6')(x) # padding to lowercase

    outputs_sequences = [net1_h_seq, net2_h_seq, net3_h_seq, net4_h_seq, net5_h_seq, net6_h_seq]
    
    return tf.keras.Model(inputs=input_tensor_sequence, outputs=outputs_sequences, name=name)


def create_encoder_cnn_model(input_shape, name="EncoderCNN"):
    """使用 Keras Functional API 創建 CNN 編碼器模型。"""
    input_tensor = tf.keras.Input(shape=input_shape, name="encoder_cnn_input")
    leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)

    # 將所有 padding="SAME" 改為 padding="same"
    conv_enc_0_out = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_0")(input_tensor)
    net1_h = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_1")(conv_enc_0_out)
    
    net2_pre_max = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="Enc_maxpool_1")(net1_h) # Fixed
    net2_h = tf.keras.layers.Conv2D(filters=43, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_2")(net2_pre_max)
    
    net3_pre_max = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="Enc_maxpool_2")(net2_h) # Fixed
    net3_h = tf.keras.layers.Conv2D(filters=57, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_3")(net3_pre_max)
    
    net4_pre_max = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="Enc_maxpool_3")(net3_h) # Fixed
    net4_h = tf.keras.layers.Conv2D(filters=76, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_4")(net4_pre_max)
    
    net5_pre_max = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="Enc_maxpool_4")(net4_h) # Fixed
    net5_h = tf.keras.layers.Conv2D(filters=101, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_5")(net5_pre_max)
    
    net6_pre_max = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same", name="Enc_maxpool_5")(net5_h) # Fixed
    net6_h = tf.keras.layers.Conv2D(filters=101, kernel_size=3, strides=1, padding="same", activation=leaky_relu, name="Enc_conv_6")(net6_pre_max)
    
    return tf.keras.Model(inputs=input_tensor, outputs=[net1_h, net2_h, net3_h, net4_h, net5_h, net6_h], name=name)


def create_decoder_model(skip_connection_shapes, final_output_channels=3, name="Decoder"):
    """使用 Keras Functional API 創建解碼器模型。"""
    input_net6_h = tf.keras.Input(shape=skip_connection_shapes[5], name="dec_input_net6_h") 
    input_net5_h = tf.keras.Input(shape=skip_connection_shapes[4], name="dec_input_net5_h")
    input_net4_h = tf.keras.Input(shape=skip_connection_shapes[3], name="dec_input_net4_h")
    input_net3_h = tf.keras.Input(shape=skip_connection_shapes[2], name="dec_input_net3_h")
    input_net2_h = tf.keras.Input(shape=skip_connection_shapes[1], name="dec_input_net2_h")
    input_net1_h = tf.keras.Input(shape=skip_connection_shapes[0], name="dec_input_net1_h")
    all_inputs = [input_net6_h, input_net5_h, input_net4_h, input_net3_h, input_net2_h, input_net1_h]
    
    leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)
    
    # 將所有 padding="SAME" 改為 padding="same"
    up1 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(input_net6_h)
    concat1 = tf.keras.layers.Concatenate()([up1, input_net5_h])
    dec_conv1_1 = tf.keras.layers.Conv2D(filters=76, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_1_1")(concat1)
    dec_conv1_2 = tf.keras.layers.Conv2D(filters=76, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_1_2")(dec_conv1_1)

    up2 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(dec_conv1_2)
    concat2 = tf.keras.layers.Concatenate()([up2, input_net4_h])
    dec_conv2_1 = tf.keras.layers.Conv2D(filters=57, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_2_1")(concat2)
    dec_conv2_2 = tf.keras.layers.Conv2D(filters=57, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_2_2")(dec_conv2_1)

    up3 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(dec_conv2_2)
    concat3 = tf.keras.layers.Concatenate()([up3, input_net3_h])
    dec_conv3_1 = tf.keras.layers.Conv2D(filters=43, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_3_1")(concat3)
    dec_conv3_2 = tf.keras.layers.Conv2D(filters=43, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_3_2")(dec_conv3_1)

    up4 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(dec_conv3_2)
    concat4 = tf.keras.layers.Concatenate()([up4, input_net2_h])
    dec_conv4_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_4_1")(concat4)
    dec_conv4_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_4_2")(dec_conv4_1)

    up5 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(dec_conv4_2)
    concat5 = tf.keras.layers.Concatenate()([up5, input_net1_h])
    dec_conv5_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_5_1")(concat5)
    dec_conv5_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=leaky_relu, name="Dec_conv_5_2")(dec_conv5_1)

    output_image = tf.keras.layers.Conv2D(filters=final_output_channels, kernel_size=3, padding="same", activation=None, name="Dec_conv_6")(dec_conv5_2) # padding to lowercase
    return tf.keras.Model(inputs=all_inputs, outputs=output_image, name=name)
