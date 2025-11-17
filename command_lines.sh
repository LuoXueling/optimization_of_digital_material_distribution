export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --thres 100 --smooth 30
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --thres 150 --smooth 30
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --thres 250 --smooth 30
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --thres 400 --smooth 30

export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 1 --thres 5000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 2 --thres 5000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 3 --thres 5000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 4 --thres 5000 --smooth 500

export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 1 --thres 4000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 2 --thres 4000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 3 --thres 4000 --smooth 500
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 4 --thres 4000 --smooth 500

export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 1 --thres 3000 --smooth 400
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 2 --thres 3000 --smooth 400
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 3 --thres 3000 --smooth 400
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 4 --thres 3000 --smooth 400
