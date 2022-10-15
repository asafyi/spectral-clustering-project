# spectral-clustering-project
Spectral Clustering implementation in C and Python using Python C-API and NumPy C-API.  
There is an interface in Python as well as a limited one in C.   
Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them.
The technique makes use of the eigenvalues of the similarity matrix of the data points to perform dimensionality reduction before clustering the data - by k-means (in this implementation).   
Various techniques including "Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)" which implemented here are discussed in the paper: [A Tutorial on Spectral Clustering.pdf](A%20Tutorial%20on%20Spectral%20Clustering.pdf)  
The project requirements: [project description.pdf](project%20description.pdf)

## C Interface
### How to build
```
bash comp.sh
```
### How to execute
```
./spkmeans <goal> <input_file>
```
**goal (enum):** Can get the following values:
- wam: Calculate and output the Weighted Adjacency Matrix.
- ddg: Calculate and output the Diagonal Degree Matrix.
- lnorm: Calculate and output the Normalized Graph Laplacian.
- jacobi: Calculate and output the eigenvalues and eigenvectors.

**input_file:**    
The path to the Input file, it will contain N data points for all
above goals except Jacobi, in case the goal is Jacobi the input is a symmetric
matrix, the file extension is .txt or .csv.

## Python Interface
### Requirements
 - numpy
### How to build
```
python3 setup.py build_ext --inplace
```
### How to execute
```
python3 spkmeans.py <k> <goal> <input_file>
```
**k (int, < N):** Number of required clusters. If equal 0, uses the eigengap heuristic algorithm.

**goal (enum):** Can get the following values:
- spk: Perform full spectral clustering and k-means.
- wam: Calculate and output the Weighted Adjacency Matrix.
- ddg: Calculate and output the Diagonal Degree Matrix.
- lnorm: Calculate and output the Normalized Graph Laplacian.
- jacobi: Calculate and output the eigenvalues and eigenvectors.

**input_file:**    
The path to the Input file, it will contain N data points for all
above goals except Jacobi, in case the goal is Jacobi the input is a symmetric
matrix, the file extension is .txt or .csv.

**For additional information about the project and how it works use the [project description file](project%20description.pdf).**

Made by [@asafyi](https://github.com/asafyi) && [@NivZindrof](https://github.com/NivZindorf)
