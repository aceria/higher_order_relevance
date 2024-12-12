# The relevance of higher-order ties

In this repository you find code to run experiments of the paper "The relevance of higher-order ties". 
If you use our code in your research or projects, please consider citing us. 


```bibtex

@article{ceria2024relevance,
  title={The relevance of higher-order ties},
  author={Ceria, Alberto and Takes, Frank W},
  journal={arXiv preprint arXiv:2412.04584},
  year={2024}
}



## Repository organization


### Description of python files

-`randomization.py`
Utilities to randomize the labelled higher-order network.

-`resilience_nx.py`
Utilities to compute the (complementary) order contribution to the number of triangles, to the global largest connected component, and to the number of nodes with each label in the largest connected component of the network.
(Based on NetworkX library)

-`resilience_checked.py`
Utilities to compute the (complementary) order contribution to the number of triangles, to the global largest connected component, and to the number of nodes with each label in the largest connected component of the network.
(Modified method to check inconsistencies, see paper in the Supplementary Material, modified approach).

-`resilience_igraph.py`
Utilities to compute the (complementary) order contribution to the global largest connected component, and to the number of nodes with each label in the largest connected component of the network.
(Based on iGraph library).

-`joint_distribution_overlapp.py`
Utilities to compute the group composition probability of a labelled higher-order network.

-`order_relevance.py`
Utilities to compute the order contribution and order relevance to the sum of the link weights in a labelled higher-order network.

-`utils.py`
Utilities for our experiments.



