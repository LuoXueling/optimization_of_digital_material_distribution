export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --radius 0.05
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --radius 0.03
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --radius 0.01

# Replace the following --load path with the actual path to your checkpoint file from one of the previous commands
export OMP_NUM_THREADS=1 && mpirun -n 8 python main_trapezoid_dist_opt.py --radius 0.03 --load outputs/trapezoid-2026-06-09-03-45-27-radius003/state.bp

export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 1 --radius 0.03
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 2 --radius 0.03
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 3 --radius 0.03
export OMP_NUM_THREADS=1 && mpirun -n 16 python main_semisphere_dist_opt_tmc.py --case 4 --radius 0.03

export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --disable_c --I1_rhs 1.0
# Replace the following --load path with the actual path to your checkpoint file from the previous command
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --disable_rho --load outputs/clamp-2026-06-09-03-07-18-topo-only-warm-start/state_opt.bp
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --disable_c --load outputs/clamp-2026-06-09-03-07-18-topo-only-warm-start/state_opt.bp
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --load outputs/clamp-2026-06-09-03-07-18-topo-only-warm-start/state_opt.bp
