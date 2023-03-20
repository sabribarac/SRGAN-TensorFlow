import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.random.set_seed(0)
import numpy as np
tf.config.run_functions_eagerly(True)
B_BLOCKS=16

class DiscriminatorBlocks(tf.keras.layers.Layer):
    def __init__(self, filters=64, kernel_size=3, stride=2, activation=.2):
        super(DiscriminatorBlocks, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.conv = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, 
                                           strides=self.stride, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation_layer = tf.keras.layers.LeakyReLU(alpha=self.activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation_layer(x)
        
        return x


class Discriminator(tf.keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = DiscriminatorBlocks(stride=1)
        self.block2 = DiscriminatorBlocks(filters=128, kernel_size=3, stride=1)
        self.block3 = DiscriminatorBlocks(filters=128, kernel_size=3, stride=2)
        self.block4 = DiscriminatorBlocks(filters=256, kernel_size=3, stride=1)
        self.block5 = DiscriminatorBlocks(filters=256, kernel_size=3, stride=2)
        self.block6 = DiscriminatorBlocks(filters=512, kernel_size=3, stride=1)
        self.block7 = DiscriminatorBlocks(filters=512, kernel_size=3, stride=2)
        
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=.2)
        self.dense2 = tf.keras.layers.Dense(1)
        self.activation = tf.keras.activations.sigmoid
        
    
    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.lrelu(x)
        x = self.dense2(x)
        x = self.activation(x)
        
        return x


def B_residual_blocks(xi, kernel_size=3, filters=64, stride=1, b_block=True):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(xi)
    x = tf.keras.layers.BatchNormalization(momentum=.5)(x)
    
    if not b_block:
        return x
    
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=.5)(x)
    x = tf.keras.layers.Add()([xi,x])
    
    return x

class BResidualBlocks(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, filters=64, stride=1, b_block=True):
        super(BResidualBlocks, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.b_block = b_block
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                            strides=self.stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=.5)
        self.prelu = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size,
                                            strides=self.stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        if self.b_block:
            x = self.prelu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.add([inputs, x])
        return x

    
class DepthToSpace(tf.keras.layers.Layer):
    def __init__(self):
        super(DepthToSpace, self).__init__()

    def call(self, inputs):
        x = tf.nn.depth_to_space(inputs,2)
        return x

class Generator(tf.keras.Model):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=9, strides=1, padding='same',input_shape=(None,None,3))
        self.prelu1 = tf.keras.layers.PReLU()
        self.resblock1 = B_residual_blocks
        self.add1 = tf.keras.layers.Add()
        
        self.blocks_3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256,kernel_size=3, strides=1, padding='same'),
            DepthToSpace(),
            tf.keras.layers.PReLU()
        ])
        
        self.blocks_4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256,kernel_size=3, strides=1, padding='same'),
            DepthToSpace(),
            tf.keras.layers.PReLU()
        ])
        self.conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x1 = self.resblock1(x)
        for _ in range(B_BLOCKS-1):
            x1 = self.resblock1(x1)
        x1 = self.resblock1(x1, b_block=False)
        x = self.add1([x,x1])
        x = self.blocks_3(x)
        x = self.blocks_4(x)
        x = self.conv2(x)
        
        return x
        
           

class SRGAN(tf.keras.Model):
    
    def __init__(self):
        super(SRGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def VGG(self,input_shape, weights='imagenet'):
        vgg = tf.keras.applications.VGG19(include_top=False, input_shape=input_shape, weights=weights)
        vgg.trainable = False
        layers = vgg.layers[20].output
        
        model = tf.keras.Model(inputs=vgg.inputs, outputs=layers)
        
        return model

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(SRGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="discriminator_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="generator_loss")
        self.vgg = self.VGG((256,256,3))
        
    def vgg_loss(self, sr, hr):

        sr = tf.keras.applications.vgg19.preprocess_input(sr)
        hr = tf.keras.applications.vgg19.preprocess_input(hr)
        
        sr = tf.cast(sr, tf.float32)
        hr = tf.cast(hr, tf.float32)
        
        sr_features = self.vgg(sr)*(1/12.75)
        hr_features = self.vgg(hr)*(1/12.75)
        return tf.keras.losses.MeanSquaredError()(hr_features, sr_features)
        
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        
        lr, hr = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            sr = self.generator(lr, training=True)
            
            hr_out = self.discriminator(hr, training=True)
            sr_out = self.discriminator(sr, training=True)
            
            l1_loss = tf.reduce_mean(tf.abs(hr-sr))
            
            content_loss = self.vgg_loss(sr,hr) + .01*self.loss_fn(tf.ones_like(sr), sr) + l1_loss
            
            disc_loss1 = self.loss_fn(tf.ones_like(hr_out), hr_out)
            disc_loss2 = self.loss_fn(tf.zeros_like(sr_out), sr_out)
            disc_loss = disc_loss1+disc_loss2
    
        d_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
            
        g_grads = gen_tape.gradient(content_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # Update metrics
        self.d_loss_metric.update_state(disc_loss)
        self.g_loss_metric.update_state(content_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

            