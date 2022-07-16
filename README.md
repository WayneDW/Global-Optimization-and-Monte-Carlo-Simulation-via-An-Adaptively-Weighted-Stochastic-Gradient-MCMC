# Global-optimization-via-an-adaptively-weighted-stochastic-gradient-MCMC
Code for "An adaptively weighted stochastic gradient MCMC algorithm for Monte Carlo simulation and global optimization"



### Requirement
* Python 3.x
* autograd
* autograd.numpy

### Global optimization on 10 non-convex functions

The adaptively weighted scheme can outperform the vanilla alternative by almost hundreds of times in the following cases (but not limited to) and is much better the existing baselines.

<img src="/images/multil-mode_exploration.png" width="800">



| Index | Function name | Dimension  | Link |
| ------------- | ------------- | ------------- | ------------- |
|1 | Rastrigin | 20  | [link](https://en.wikipedia.org/wiki/Rastrigin_function)  |
|2 | Griewank  | 20  | [link](https://www.sfu.ca/~ssurjano/griewank.html)  |
|3 | Sum Squares | 20 | [link](https://en.wikipedia.org/wiki/Sum_of_squares_function) |
|4 | Rosenbrock  | 20 |[link](https://en.wikipedia.org/wiki/Rosenbrock_function)  |
|5 | Zakharov  | 20   |[link](https://www.sfu.ca/~ssurjano/zakharov.html)  |
|6 | Powell | 24 | [link](https://www.sfu.ca/~ssurjano/powell.html) |
|7 | Dixon & Price | 25 | [link](https://www.sfu.ca/~ssurjano/dixonpr.html)  |
|8 | Levy | 30 | [link](https://www.sfu.ca/~ssurjano/levy.html) |
|9 | Sphere | 30 | [link](https://www.sfu.ca/~ssurjano/spheref.html) |
|10 | Ackley | 30 | [link](https://www.sfu.ca/~ssurjano/ackley.html) |


#### How to run the algorithms

Test (1) Rastrigin function
```python
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method awsgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method   sgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
```

Test (2) Griewank function
```python
>> python3 ./main.py -fnum 2 -lr 0.1  -T 10 -error 25 -method awsgld -div 5 -part 100 -zeta 10
>> python3 ./main.py -fnum 2 -lr 0.1  -T 10 -error 25 -method   sgld -div 5 -part 100 -zeta 10
```


Test (3) Sum Squares function
```python
>> python3 ./main.py -fnum 3 -lr 0.01 -T 0.01 -error 1.5 -method awsgld -div 1 -part 100 -zeta 1
>> python3 ./main.py -fnum 3 -lr 0.01 -T 0.01 -error 1.5 -method   sgld -div 1 -part 100 -zeta 1
```


Test (4) Rosenbrock function
```python
>> python3 ./main.py -fnum 4 -lr 1e-5 -T 10 -error 20 -method awsgld -div 3 -part 100 -zeta 10
>> python3 ./main.py -fnum 4 -lr 1e-5 -T 10 -error 20 -method   sgld -div 3 -part 100 -zeta 10
```


Test (5) Zakharov function
```python
>> python3 ./main.py -fnum 5 -lr 1e-9 -T 10000 -error 500 -method awsgld -div 50 -part 100 -zeta 0.5
>> python3 ./main.py -fnum 5 -lr 1e-9 -T 10000 -error 500 -method   sgld -div 50 -part 100 -zeta 0.5
```

Test (6) Powell function
```python
>> python3 ./main.py -fnum 6 -lr 1e-4 -T 1 -error 1 -method awsgld -div 2 -part 100 -zeta 200
>> python3 ./main.py -fnum 6 -lr 1e-4 -T 1 -error 1 -method   sgld -div 2 -part 100 -zeta 200
```

Test (7) Dixon & Price function
```python
>> python3 ./main.py -fnum 7 -lr 1e-5 -T 10 -error 10 -method awsgld -div 2 -part 100 -zeta 20
>> python3 ./main.py -fnum 7 -lr 1e-5 -T 10 -error 10 -method   sgld -div 2 -part 100 -zeta 20
```

Test (8) Levy function
```python
>> python3 ./main.py -fnum 8 -lr 1e-4 -T 100 -error 400 -method awsgld -div 60 -part 100 -zeta 10
>> python3 ./main.py -fnum 8 -lr 1e-4 -T 100 -error 400 -method   sgld -div 60 -part 100 -zeta 10
```

Test (9) Sphere function
```python
>> python3 ./main.py -fnum 9 -lr 1e-2 -T 1e-4 -method awsgld -div 2 -part 100 -zeta 1
>> python3 ./main.py -fnum 9 -lr 1e-2 -T 1e-4 -method   sgld -div 2 -part 100 -zeta 1
```

Test (10) Ackley function
```python
>> python3 ./main.py -fnum 10 -lr 0.01 -T 0.05 -error .4 -check 1 -method awsgld -div 0.04 -part 100 -zeta 0.2
>> python3 ./main.py -fnum 10 -lr 0.01 -T 0.05 -error .4 -check 1 -method   sgld -div 0.04 -part 100 -zeta 0.2
```

### Mode explorations on MNIST dataset
