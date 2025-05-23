# Learning-Topic-Hierarchy

This is the main code for "Learning Topic Hierarchies by Tree-Directed Latent Variable Models" [S. Chakraborty, R. Lei, X. Nguyen] (https://www.arxiv.org/abs/2408.14327)

Look at the demo.ipynb notebook for details regarding the parts of the code - main files are in DRT folder. The NYT folder contains the New York Times data along with pre-processing steps.

Main code py files:
1. generation - setting up helper functions for generating synthetic data
2. tree_structure - implements utility functions for tree structures
3. initializers - various initialization techniques (used in the fit function)
4. sampler_LDA - collapsed Gibbs sampler for usual Latent Dirichlet Allocation
5. sampler_DRT - collapsed Gibbs sampler for the tree-structured topic model as discussed in the paper
6. simulation_utils - other helper functions, including fit function which uses 8 initializations to run parallel chains
7. metrics - computes useful metrics useful for the simulation studies

Library versions:
1. jax '0.4.23'
2. jaxlib '0.4.23.dev20240124'
3. numpy '1.26.3'
4. scikit-learn '1.3.0'
5. scipy '1.11.4'
6. pot '0.8.2' (optimal transport library)
7. cvxpy '1.4.1' (convex optimization library)

The last three are only used for the metrics.

For code related questions, contact: Sunrit Chakraborty (sunritc@umich.edu)
