# PBI
The code related to the paper below：

Pairwise-interactions-based Bayesian Inference of Network Structure from Cascades

# Running the code:
1.  install the required packages
2.  run the test.py to infer the network
3.  we provide a simple dataset: a ground-truth network generated by ER random graph model ( nodes = 253, edges = 510) , 
4.  G.pkl is a file that stores a network object from networkx package and can be read using the pickle package.
5.  "beta 500.txt" is a file that stores 500 cascades generated by the IC model （incubation time ~ exp(1)）on this network.
