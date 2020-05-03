# FD-Solution-to-Poisson-PDE-with-HDBCs
A finite difference algorithm which computes a numerical solution to the Poisson partial differential equation in two real variables on a cell with homogeneous Dirichlet boundary conditions.
This code implements a finite difference scheme (specifically, the 5-point stencil shown here https://en.wikipedia.org/wiki/Finite_difference_method#Example:_The_Laplace_operator)
to approximate a solution u (a real-valued function of two real variables) on a cell [a,b]x[c,d] to the Poisson equation 
"f(x,y) is negative Laplacian of u(x,y) for (x,y) in (a,b)x(c,d)" (where f is known, and is also a real-valued function of two real variables) 
subject to the Dirichlet boundary condition that "(x=a or x=b or y=c or y=d) implies u(x,y)=0."

For the implementation, scipy.sparse and scipy.sparse.linalg are relied on. The (criminally underated) package Plotly is used to generate surface plots, and the package tqdm is used to time performance.
