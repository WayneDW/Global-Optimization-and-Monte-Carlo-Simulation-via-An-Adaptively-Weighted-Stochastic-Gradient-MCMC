# Global-optimization-via-an-adaptively-weighted-stochastic-gradient-MCMC
Code for "An adaptively weighted stochastic gradient MCMC algorithm for Monte Carlo simulation and global optimization"



### Requirement
* Python 3.x
* numpy



The adaptively weighted Langevin dynamics can outperform the vanilla alternative by almost hundreds of times in the following cases (but not limited to).


| Index | Dimension | Function name  | Link |
| ------------- | ------------- | ------------- | ------------- |
|1 | 20 | Rastrigin  | [link](https://en.wikipedia.org/wiki/Rastrigin_function)  |
|2 | 20 | Griewank  | [link](https://www.sfu.ca/~ssurjano/griewank.html)  |
|3 | 20 | Sum Squares | [link](https://en.wikipedia.org/wiki/Sum_of_squares_function) |
|4 | 20 | Rosenbrock  | [link](https://en.wikipedia.org/wiki/Rosenbrock_function)  |
|5 | 20 | Zakharov  | [link](https://www.sfu.ca/~ssurjano/zakharov.html)  |
|6 | 24 | Powell | [link](https://www.sfu.ca/~ssurjano/powell.html) |
|7 | 25 | Dixon & Price | [link](https://www.sfu.ca/~ssurjano/dixonpr.html)  |
|8 | 30 | Levy | [link](https://www.sfu.ca/~ssurjano/levy.html) |
|9 | 30 | Sphere | [link](https://www.sfu.ca/~ssurjano/spheref.html) |
|10 | 30 | Ackley | [link](https://www.sfu.ca/~ssurjano/ackley.html) |


### How to run the algorithms

Run AWSGLD
```
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method awsgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
```

Run SGLD
```
>> python3 ./main.py -fnum 1 -lr 5e-4 -T 5 -error 75 -check 1 -method sgld -div 3 -part 100 -zeta 0.02 -decay_lr 200
```
