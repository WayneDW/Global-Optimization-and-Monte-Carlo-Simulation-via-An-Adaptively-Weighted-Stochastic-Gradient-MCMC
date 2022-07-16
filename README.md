# Global-optimization-via-an-adaptively-weighted-stochastic-gradient-MCMC
Code for "An adaptively weighted stochastic gradient MCMC algorithm for Monte Carlo simulation and global optimization"



### Requirement
* Python 3.x
* numpy

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


### How to run the algorithms

Run AWSGLD
```
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method awsgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
```

Run SGLD
```
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method sgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
```

The hyperparameters are detailed hyperparameter_part${N} files, where N denotes the number of (10 or 100) partitions.
