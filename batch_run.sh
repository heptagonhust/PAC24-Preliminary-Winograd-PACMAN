#!/bin/bash
#SBATCH -p pac
#SBATCH -n 1             
#SBATCH --exclusive
#SBATCH -c 160
#SBATCH --output=/share/org/PARATERA/PAC20241694/fyf/wino_final/Output_winograd

#================ you can configure the sbatch option above ===============



# ================ configure your own parameters here =====================
export OMP_NUN_THREADS=160
export OMP_PROC_BIND=True
export OMP_PLACES=cores 
# =========================================================================

echo "$(date '+%Y-%m-%d %H:%M:%S') --- User : ${user_name}" 

# execute the program
# ./winograd ${path_to_casefile} ${test_mode}
if  [ $test_mode -eq 0 ];then
    for i in {1..3}            # test 10 times 
        do
         ./winograd ${path_to_casefile} 0
        if  [ $i -eq 3 ];then 
            perf stat -e cpu-cycles,instructions,cache-references,cache-misses,context-switches,major-faults,minor-faults --  ./winograd ${path_to_casefile} ${test_mode}
        fi
        done
else
     ./winograd ${path_to_casefile} 1
fi
