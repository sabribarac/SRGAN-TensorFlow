# Single Image Super-Resolution with SRGAN
Single Image Super-Resolution build on TensorFlow 2.11.0.

The code is based on the following paper,
  - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

------------------------
## Results
After some epochs this was the result (dataset from **Set 5**):

![[](/image_SRF_4/srgan_result.png?raw=true)](https://github.com/sabribarac/srgan/blob/3fea85528f9dbeb6c000f44363bafbd52c862192/image_SRF_4/srgan_result.png)
## How to use

In a few simple lines you can train the model,

```
from srgan import SRGAN

# Instantiate SRGAN model
srgan = SRGAN()

# Define optimizers and loss function
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Compile the model
srgan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer, loss_fn=loss_fn)

# Train the model
srgan.fit(dataset, epochs=150)
```
