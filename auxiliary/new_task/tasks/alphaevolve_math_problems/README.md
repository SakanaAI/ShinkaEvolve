## AlphaEvolve mathematical problems

This folder contains the necessary evaluator and initial program files for all of the 14 problems from [AlphaEvolve's Appendices A and B](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf). The circle packing problem was among the first implemented by OpenEvolve. This folder implements the remaining 13 problems. 

Not all problems take the form of a maximization problem, however in order to make this problem set more standardized we chose to **make it so all evaluator files are aiming to maximize the target metric**. We achieve by some straightforward algebraic manipulation, but this can be easily edited and changed if the user finds it necessary.

The problem from Appendix A is the following:
- [Matrix multiplication](matmul): obtain a faster algorithm for multiplying two matrices of sizes $m \times n$ and $n \times p$ (c.f. Appendix A).

The remaining problems are from Appendix B:
1. [First autocorrelation inequality](first_autocorr_ineq): Construct a nonnegative step function $f:\mathbb{R} \mapsto \mathbb{R}$ to improve an upper bound on a constant related to the autoconvolution of $f$ (c.f. Appendix B.1.).
2. [Second autocorrelation inequality](second_autocorr_ineq): Construct a nonnegative step function $f:\mathbb{R} \mapsto \mathbb{R}$ to improve a lower bound on a constant related to the norm of the autoconvolution of $f$ (c.f. Appendix B.2.).
3. [Third autocorrelation inequality](third_autocorr_ineq): Construct a nonnegative step function $f:\mathbb{R} \mapsto \mathbb{R}$ to improve an upper bound on a constant related to th absolute value of the autoconvolution of $f$ (c.f. Appendix B.3.).
4. [An uncertainty inequality](uncertainty_ineq): Construct a function $f:\mathbb{R} \mapsto \mathbb{R}$ to obtain an upper bound on a constant related to $f$ and its fourier transform. (c.f. Appendix B.4.).
5. [Erdos minimum overlap problem](erdos_min_overlap): Construct a nonnegative step function $f:\mathbb{R} \mapsto \mathbb{R}$ satisfyind some special properties to improve an upper bound on a constant that controls the asymptotics of the Minimum Overlap Problem (c.f. Appendix B.5.).
6. [Sums and differences of finite sets](sums_diffs_finite_sets): Construct a set of nonnegative integers $U$ satisfying some special properties to improve a lower bound to a constant related to sums and differences of finite sets (c.f. Appendix B.6.).
7. [Packing unit regular hexagons inside a regular hexagon](hexagon_packing): Place $n$ disjoint unit regular hexagons inside a larger regular hexagon, minimizing the side length of the outer hexagon. We consider the case where $n = 11$ and $n = 12$ (c.f. Appendix B.7.).
8. [Minimizing the ratio of maximum to minimum distance](minimizing_max_min_dist): Place $n$ $d$-dimensional points in order to minimize the ratio between the maximum and minimum pairwise distances. We consider the cases where $n=16,d=2$ and $n=14,d=3$ (c.f. Appendix B.8.).
9. [The Heilbronn problem for triangles](heilbronn_triangle): Place $n$ points on or inside a triangle with unit area so that the area of the smallest triangle formed by these points is maximized. We consider the case where $n = 11$ (c.f. Appendix B.9.).
10. [The Heilbronn problem for convex regions](heilbronn_convex): Place $n$ points on or inside a convex region with unit area so that the area of the smallest triangle formed by these points is maximized. We consider the case where $n = 13$ and $n=14$ (c.f. Appendix B.10.).
11. [Kissing number in dimension 11](kissing_number): Increase the lower bound on the $11$-dimensional kissing number, i.e., the number of disjoint unit spheres that can be packed tangent to a given unit sphere (c.f. Appendix B.11.).
12. [Packing circles inside a unit square to maximize sum of radii](../circle_packing): Place $n$ disjoint circles inside a unit square so as to maximize the sum of their radii (c.f. Appendix B.12.).
13. [Packing circles inside a rectangle of perimeter 4 to maximize sum of radii](circle_packing_rect):  Place $n$ disjoint circles inside a rectangle of perimeter $4$ so as to maximize the sum of their radii (c.f. Appendix B.13.).
