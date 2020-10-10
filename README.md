
Code used the implement the mean-field Ising Influence Maximization algorithm at both the level of a full network and at the level of blocks within a graph with block structure.

![](https://github.com/MGarrod1/ising_block_level_influence/blob/master/example/full_control_on_graph.png)

Plot showing the estimate of the mean-field optimal control field on a two block SBM.

## Requirements

## Installation

### Applications

This module is used for the numerical simulations in the paper:

Garrod M., and N. S. Jones. Influencing social networks without knowledge of
network micro-structure. In preparation, 2020.

This code was also used for the numerical simulations in my PhD thesis *"Influence and ensemble variability in unobserved networks"* (available at: [https://doi.org/10.25560/83107](https://doi.org/10.25560/83107)). The simulations used an earlier version fo this code base, however, the implementations of the algorithms used are essentially the same.

### References

The implementation of the Ising influence maximisation algorithm builds upon ideas described in the paper: 

Lynn, Christopher, and Daniel D. Lee. "Maximizing influence in an ising network: A mean-field optimal solution." Advances in Neural Information Processing Systems. 2016. 

The code used for the projection onto the positive simplex is based on the ideas presented in the paper:

Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex
Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
ICPR 2014.

and can be found at: [https://gist.github.com/mblondel/6f3b7aaad90606b98f71](https://gist.github.com/mblondel/6f3b7aaad90606b98f71)


