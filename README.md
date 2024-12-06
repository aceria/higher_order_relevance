# The relevance of higher-order ties

In this repository you find code to run experiments of the paper "The relevance of higher-order ties". 



## Repository organization

### Description of folders

- [code/](code/): Rutines to run the core experiments.

### Description of python files

-`randomization.py`
Utilities to randomize the labelled higher-order network.

-`resilience_nx.py`
Utilities to compute the (complementary) order contribution to the number of triangles, to the global largest connected component, and of the number of nodes at each label in the largest connected component of the network.
(Uses NetworkX library)

-`resilience_checked.py`
Utilities to compute the (complementary) order contribution to the number of triangles, to the global largest connected component, and of the number of nodes at each label in the largest connected component of the network.
(Modified method to check inconsistencies, see paper in the Supplementary Material, modified approach).

-`resilience_igraph.py`
Utilities to compute the (complementary) order contribution to the global largest connected component, and of the number of nodes at each label in the largest connected component of the network.
(Uses NetworkX library).

-`joint_distribution_overlapp.py`
Utilities to compute the group composition probability of a labelled higher-order network.

-`order_relevance.py`
Utilities to compute the order contribution and order relevance to the sum of the link weights in a labelled higher-order network.

-`utils.py`
Utilities for our experiments.



