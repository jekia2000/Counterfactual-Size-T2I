#!/bin/bash
source /leonardo_work/EUHPC_B25_058/aleksa_files/downloaded_work/aj/bin/activate
salloc --job-name=setup \
       --account=EUHPC_B25_058 \
       --partition=boost_usr_prod \
       --qos=boost_qos_dbg \
       --time=00:30:00 \
       --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=3 \
       --mem-per-cpu=5000M \
       --gres=gpu:1 \
       --mail-type=BEGIN,END,FAIL \
       --mail-user=aleksa.jelaca@student.kuleuven.be
