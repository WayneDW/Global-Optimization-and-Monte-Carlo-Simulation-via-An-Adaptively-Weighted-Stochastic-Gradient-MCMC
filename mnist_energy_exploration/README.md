# AWSGLD

### Mode explorations on MNIST dataset

### Requirement
* Python 3.x

Although MNIST has been talked about a billion times, the MCMC algorithms cannot achieve free exploration / fluctuating losses using a fixed learning rate. Luckily, such a tragedy has been solved through this code.

<img src="mnist_energy_exploration/images/mnist_mode_exploration.png" width="800">

AWSGHMC
```python
>> python bayes_cnn.py -c awsghmc -lr 1e-7 -zeta 3e4 -T 1
```

For other baselines, you can follow:

Preconditioned SGLD
```python
>> python bayes_cnn.py -c psgld -zeta 0 -T 0.3 -lr 3e-7
```

SGHMC

```python
>> python bayes_cnn.py -c sghmc -zeta 0 -T 0.3 -lr 1e-6
```


Cyclical SGHMC
```python
>> python bayes_cnn.py -c cyc -zeta 0 -T 0.1 -lr 2e-6 
```

