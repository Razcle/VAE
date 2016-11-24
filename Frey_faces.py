"""
 This Experiment applies the variational auto-encoder to the Frey face data-set
 and records the lower-bound as a function of iteration number. It also saves
 the final model and generates a sample of a face.
"""
import utilities as util
from matplotlib import pyplot as plt
import numpy as np

data, _ = util.load_frey_faces(0.0)

# Create a VAE and Train
network_architecture = {"hdn_dim": 500, 'latent_dim': 2, 'input_dim': 560}
vae_2d = util.train(data, network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=100, display_step=1)

# Generate and show a faces in the latent space
nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 20*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]])
        x_mean = vae_2d.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*20:(j+1)*20] = x_mean[0].reshape(28, 20)

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper")
plt.figure(figsize=(7, 5))
plt.show()
vae_2d.sess.close()
