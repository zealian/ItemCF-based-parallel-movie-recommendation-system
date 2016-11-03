This program is aiming to use Item-based Collaborative Filtering to do movie recommendation. The source data (MovieLens 1M dataset) of movie can be downloaded from http://grouplens.org/datasets/movielens/. The algorithm calculating similarity is pearson correlation.

MPI and openmp are applied to improve the performance.

Compile the code:
mpic++ itemcf.cpp -fopenmp

Example script:
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

mpirun -np 2 ./a.out 0 10 1 2 3 4

Command line arguments:
mode top-k uid1 [uid2...]

When first run the program, must use mode 0 to generate offline similarity file.

Explanation of mode:
Mode 0: online mode, used to update similarity or generate new similarity file, then use the new similarity file to give recommendations.

Mode 1: offline mode, use similarity file generated before to give recommendation. This one is much more faster than mode 0, because no need to calculate similarity between movies.