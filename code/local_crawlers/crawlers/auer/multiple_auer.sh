#!/bin/bash

if [ $# -lt 11 ]; then
	echo "Usage: $0 <similarity_threshold> <m> <w> <budget:-1 for unlimited> <n in n_grams used for DOM path vector representation> <batch_size_url> <use_url_classifier:1 for True> <alpha (2s2 for 2\sqrt{2})> <log_path_all_sites> <num_executions> <site_name1> <site_name2> ... <site_namek>"
	exit 1
fi
threshold=$1
m=$2
w=$3
budget=$4
n=$5
batch_size_url=$6
use_url_classifier=$7
alpha=$8
log_path_all_sites=$9
num_executions=${10}
shift 10
site_names=("$@")

for site_name in "${site_names[@]}"; do
	echo "Launching $num_executions runs for site $site_name ..."
	for (( i=1; i<=$num_executions; i++ )); do
		python3 local_crawlers/auer/auer.py "$threshold" "$m" "$w" "$budget" "$n" "$batch_size_url" "$use_url_classifier" "$alpha" "$site_name" "$log_path_all_sites" &
		sleep 4
	done
	wait
	echo "Done."
done

wait
echo "All local crawlings finished."