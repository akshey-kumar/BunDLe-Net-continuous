import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
import cebra.datasets
from functions import Database, preprocess_data, prep_data, plotting_neuronal_behavioural, plot_latent_timeseries, plot_phase_space, rotating_plot
from bundlenet import BunDLeNet, train_model

sys.path.append(r'../')

### Load Data and plot
rat_names = ['achilles', 'gatsby','cicero', 'buddy']
rat_name = rat_names[2] 
hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-' + rat_name)

X = hippocampus_pos.neural.numpy()
B = hippocampus_pos.continuous_index.numpy()

### Prepare data for BundLe Net
X_, B_ = prep_data(X, B, win=20)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

loss_array = train_model(
	X_,
	B_,
	model,
	optimizer,
	gamma=0.9, 
	n_epochs=2000,
	pca_init=False
)

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

algorithm = 'BunDLeNet'
### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_rat_' + str(rat_name))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_rat_' + str(rat_name), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_rat_' + str(rat_name), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_rat_' + str(rat_name))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_rat_' + str(rat_name)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y0_, B_, state_names)
plot_phase_space(Y0_, B_, state_names = state_names)
rotating_plot(Y0_, B_,filename='figures/rotation_'+ algorithm + '_rat_'+str(rat_name) +'.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

### Recurrence plot analysis of embedding
pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
plt.matshow(pd_Y, cmap='Greys')
plot_latent_timeseries(Y0_, B_, state_names)
plt.show()

