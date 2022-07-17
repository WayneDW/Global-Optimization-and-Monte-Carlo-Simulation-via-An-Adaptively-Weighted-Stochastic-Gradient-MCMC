# AWSGLD

### Mode explorations on MNIST dataset

### Requirement
* Python 3.x

Although MNIST has been talked about a billion times, the MCMC algorithms cannot achieve free exploration / fluctuating losses using a fixed learning rate. Luckily, such a tragedy has been solved through this code.

<img src="/mnist_energy_exploration/images/mnist_mode_exploration.png" width="800">

AWSGHMC
```python
>> python bayes_cnn.py -c awsghmc -lr 1e-7 -T 1 -zeta 3e4
```

For other baselines, you can follow:

Preconditioned SGLD
```python
>> python bayes_cnn.py -c psgld -lr 3e-7 -T 0.3
```

SGHMC

```python
>> python bayes_cnn.py -c sghmc -lr 1e-6 -T 0.3
```


Cyclical SGHMC
```python
>> python bayes_cnn.py -c cyc -lr 2e-6 -T 0.1
```

