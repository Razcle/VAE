"""
 This Experiment applies the variational auto-encoder to the Frey face data-set
 and records the lower-bound as a function of iteration number. It also saves
 the final model and generates a sample of a face.
"""

import utilities as util
import numpy as np
from matplotlib import pyplot as plt

# Get the Data
train, test = util.load_frey_faces()

# Create a VAE and Train
network_architecture = {"hdn_dim": 20, 'latent_dim': 2, 'input_dim': 560}
vae_2d = util.train(train, network_architecture)

nx = ny = 20
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 20*nx))
for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
        z_mu = np.array([[xi, yi]])
        x_mean = vae_2d.generate(z_mu)
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_values, y_values)
plt.imshow(canvas, origin="upper")
plt.tight_layout()
plt.show()
