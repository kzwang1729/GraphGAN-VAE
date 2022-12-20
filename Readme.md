# GraphGAN-VAE

The following is code for our new novel architecture, GraphGAN-VAE. GraphGAN-VAE is a deep generative model for graph generation, employing a GAN architecture with a VAE generator to iteratively generate nodes from a starting graph seed. Both the generator (encoder + decoder) and the discriminator are implemented using GCN layers (Kipf & Welling, 2016).

We train the model on three different datasets (cora, citeseer, ENZYME) to evaluate the generated graphs using metrics proposed in GraphRNN

# Execution

To setup the enviornment, simply execute 
```
pip install -r requirements.txt
```

Then, run 
```
run.py
```

