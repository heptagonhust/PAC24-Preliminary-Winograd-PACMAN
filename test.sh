#!/bin/bash


if [ -z $1  ] || [ -z $2 ] || [ -z $3 ]; then
	echo "Usage: $0 <case_file_path> <test_mode> <user_name>"
    echo "Example : $0 small.cnf 0 jpx "
	exit 1
fi

data_path=/share/org/PARATERA/PAC20241694/Output_wingorad
search_exec_path=/share/org/PARATERA/PAC20241694/$3
# echo ${search_exec_path}
if [ ! -d ${data_path} ]; then    
		mkdir -p ${data_path}
fi

if [ ! -d ${search_exec_path} ]; then    
		echo "wrong user_name_dir"
        exit 1
fi


exec_path=$(find ${search_exec_path} -name "Makefile" -exec pwd {} \;)
case_file_path=$(find ${search_exec_path} -name "Makefile" -exec pwd {} \; )

export path_to_casefile=$(pwd)/$1
export test_mode=$2
export user_name=$3
echo ${path_to_casefile}
echo ${exec_path}

# pushd ${exec_path}
# make clean

# available module 
# this is the newest version (3.2.0) older version you could check with cmd 'module avail'
# bisheng/3.2.0-aarch64  mpi/hmpi/2.3.0-bisheng3.2.0-aarch64 
# libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64 

# module load
# ================== Configure your modules here ======================

module load bisheng/3.2.0-aarch64

module load libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64 

# ======================================================================


make
sbatch ./batch_run.sh
# popd

echo "Program exits. Check the output file in /share/org/PARATERA/PAC20241694/Output_wingorad"

# module unload
# ======= don't forget to unload the module you've loaded before ===========
module unload bisheng/3.2.0-aarch64
module unload libs/kml/2.2.0-bisheng3.2.0-hmpi2.3.0-aarch64 

# ===========================================================================