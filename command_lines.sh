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

export OMP_NUM_THREADS=1 && mpirun -n 4 python main_compress_ball_dist_opt_tmc.py --case dist
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_compress_ball_dist_opt_tmc.py --case topo

export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo --lr_c 0.0 --lr_rho 50.0

export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --lr_c 50.0 --lr_rho 0.0 --max_penalty 6.0 --load outputs/clamp-2026-02-12-11-25-26/state_opt.bp
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --lr_c 0.0 --lr_rho 10.0 --max_penalty 6.0 --load outputs/clamp-2026-02-12-11-25-26/state_opt.bp
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1 --lr_c 50.0 --lr_rho 10.0 --max_penalty 6.0 --load outputs/clamp-2026-02-12-11-25-26/state_opt.bp

export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m topo
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m topo-i1
export OMP_NUM_THREADS=1 && mpirun -n 4 python main_clamp.py -m dist-topo-i1

export OMP_NUM_THREADS=1 && mpirun -n 64 python main_clamp.py -m dist-topo -c ./mesh/config_clamp_3d.py  --lr_c 0.0 --lr_rho 50.0

python main_cantilever_beam.py
