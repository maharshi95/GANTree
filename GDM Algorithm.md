```
max_iter_gen = 5
max_iter_disc = 20

disc_accuracy_limit = 95
gen_accuracy_limit = 60
```

```
Procedure: SingleNodeGDM
Inputs: node

REPEAT:
    FOR j in {1, ..., n} do:
        sample x from Dataset(node)
        sample z from Prior(node)
        
        // Generator and Encoder Training
        
        E, G, D = node.networks()
        
        L_recon = ||x - G(E(x))|| + ||z - E(G(z))||
        L_adv_gen = -log(D(G(z)))
        
        theta(enc), theta(dec) = Adam(theta(enc), theta(dec), L_recon + lambda * L_adv_gen)
        
        FOR k in {1, ..., m} do:
            
            // Decoder Training
        
            L_adv_dis = -log(D(x)) - log(1 - D(G(z)))
            theta(disc) = Adam(theta(disc), L_disc_adv)
```

```
Procedure: MultiNodeGDM
Input Node Index i:

REPEAT till saturation:
    
    sample x from Pr(left(i))
    sample z from P(left(i))
    
    // Generator and Encoder Training
    
    L_recon = ||x - G(E(x))|| + ||z - E(G(z))||
    L_adv_gen = -log(D(G(z)))
    
    theta(enc), theta(dec) = Adam(theta(enc), theta(dec), L_recon + lambda * L_adv_gen)
    
    FOR k in {1, ..., m} do:
        
        // Decoder Training
    
        L_adv_dis = -log(D(x)) - log(1 - D(G(z)))
        theta(disc) = Adam(theta(disc), L_disc_adv)
```

```
Procedure: ModeSeperationMethod
Inputs   : node (GAN-Node)

REPEAT untill saturation:
    X ~ dataset(node.id)
    
    label
    
    L_Xrecon = ||x - G(E(x))||
    L_CE = - log (MAX(P(mode|x))
    L_reg = Tr(Sigma_sv)
    
    L = L_Xrecon + lambda_ce * L_CE + lambda_reg * L_reg
    
    theta(E, D) = Adam(theta(E, D), L)
    
```


```
Create root Node with id 0

SingleNodeGDM(root)

REPEAT till saturation:

    Z = node.E(x)
    
    Z1, Z2 = KMeans(Z)
    
    Fit two Gaussian Mixture on node.E(X)
    
    // Split Node
    E = node.E.copy()
    G_left = node.G.copy()
    D_left = node.D.copy()
    
    G_right = node.G.copy()
    D_right = node.D.copy()
    
    lnode = create_new_node(E, G_left, D_left)
    rnode = create_new_node(E, G_right, D_right)
       
    // Mean, Covariances and Prior Probability
    lnode.dist_params = mean(Z1), covariance(Z1), |Z1|/|Z|
    rnode.dist_params = mean(Z2), covariance(Z2), |Z2|/|Z|
    
    left(node) = lnode
    right(node) = rnode
    
    for i in {1, ..., k1}:
        ModeSeparationMethod(node)
    
    for i in {1, ..., k2}:
        MultiNodeGDM(node)

```