from keras.layers import Dense,Input,Lambda
from keras import Model
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.utils import plot_model



def simpleAE(input_shape, label_num, hidden_dim=128, encoder_dim=16):
    # encoder
    inputs = Input(shape=input_shape)
    hidden_layer1 = Dense(hidden_dim, activation='relu')(inputs)
    hidden_layer2 = Dense(hidden_dim, activation='relu')(hidden_layer1)
    encoded = Dense(encoder_dim, activation='relu')(hidden_layer2)
    encoder = Model(inputs= inputs, outputs= encoded)

    # decoder
    hidden_layer3 = Dense(hidden_dim, activation='relu')(encoded)
    decoded = Dense(label_num, activation='tanh')(hidden_layer3)
    decoder = Model(inputs=encoded, outputs=decoded)

    # auto-encoder
    autoencoder = Model(inputs=inputs, outputs=decoded)
    autoencoder.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

    return encoder, decoder, autoencoder


def VAE(original_dim, hidden_dim=128, encoder_dim=16):
    # encoder
    inputs = Input(shape=(original_dim,))
    hidden_layer1 = Dense(hidden_dim, activation='relu')(inputs)
    encoded = Dense(encoder_dim, activation='relu')(hidden_layer1)
    z_mean = Dense(encoder_dim)(encoded)
    z_log_var = Dense(encoder_dim)(encoded)
    #使用均值变量（mean vector）和标准差变量（standard deviation vector）合成隐变量
    def sampling(args):
        z_mean, z_log_var = args
        #使用标准正态分布初始化
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], hidden_dim), mean=0.,stddev=1.0)
        #合成公式
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(hidden_dim,))([z_mean, z_log_var])
    encoder = Model(inputs=inputs, outputs=z)
    encoder.summary()

    # decoder
    hidden_layer2 = Dense(hidden_dim, activation='relu')(z)
    decoded = Dense(original_dim, activation='tanh')(hidden_layer2)
    decoder = Model(inputs=encoded, outputs=decoded)
    decoder.summary()
    
    # vae
    vae = Model(inputs, decoded)
    reconstruction_loss = original_dim * binary_crossentropy(inputs,
                                                decoded)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)
