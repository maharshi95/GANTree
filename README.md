# GANTree

A hierarchical tree based architecture over Generative Adversarial Networks (GANs) for generation of samples from multi-modal distributions through discontinuous embedding manifold, having a flexibility to tweak the degree of interpolatability across the modes in the latent space.

```

GAN Tree Algorithm:

X = {0: all_x_data}

CREATE node0:
    node0.E = Encoder()
    node0.D = Decoder()
    node0.Di = Disc()
    
    node0.mu = [0, 0]
    node0.cov = [[1, 0], [0, 1]]

for node0: for each step:
    TRAIN node0.E and node0.D over cyclic_loss
    TRAIN node0.Di over disc_adv loss
    TRAIN node0.D over gen_adv loss
    
node0.gmm = GaussianMixture()
node0.gmm.fit(node0.encode(X[0]))
split X into 2 labels: 1 and 2

CREATE node1 and node2 from node0:
    E = node0.E.copy()
    node1.E = E
    node2.E = E
    
    node1.D = node0.D.copy()
    node2.D = node0.D.copy()
    
    node1.Di = node0.Di.copy()
    node2.Di = node0.Di.copy()
    
    node1.mu = node0.gmm.means_[0]
    node2.mu = node0.gmm.means_[1]
    
    node1.cov = node0.gmm.cov_[0]
    node2.cov = node0.gmm.cov_[1]

FOR k iterations:
    repeat for a iters {
        node = sample(node1, node2)
        
        z ~ N(node.mu, node.cov)
        x ~ X[node.id]
        
        TRAIN node.E and node.D over cyclic_loss
        TRAIN node.Di over disc_adv loss
        TRAIN node.D over gen_adv loss
        
        TRAIN node.E over x_clf_loss
    }
    z = [node1.E(X[1]); node2.E(X[2])] # mix them
    gmm = GaussianMixture()
    gmm.fit(z)
    relabel X[0] with 1 and 2 and split

```