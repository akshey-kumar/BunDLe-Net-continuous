from tensorflow.keras import layers, Model
import tensorflow as tf

### Autoencoder architecture
class Autoencoder(Model):
    '''
    Autoencoder module for neuronal data reconstruction
    latent_dim should be set to 3 for comparision with BunDLe-Net
    ninput_shape should be set to always X0_.shape
    '''
    def __init__(self, latent_dim, ninput_shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.ninput_shape = ninput_shape
        self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(50, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dense(25, activation='relu'),
                layers.Dense(10, activation='relu'),
                layers.Dense(latent_dim, activation='linear'),
        ])
        self.decoder = tf.keras.Sequential([
                layers.Dense(latent_dim, activation='relu'),
                layers.Dense(10, activation='relu'),
                layers.Dense(25, activation='relu'),
                layers.Dense(30, activation='relu'),
                layers.Dense(50, activation='relu'),
                layers.Dense(self.ninput_shape[-1]*self.ninput_shape[-2], activation='linear'),
                layers.Reshape(self.ninput_shape[1:])
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded