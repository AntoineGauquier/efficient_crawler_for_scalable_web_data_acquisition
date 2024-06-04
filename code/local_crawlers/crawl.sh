#!/bin/bash
clear
echo "Select the crawler you want to use for local crawling:"
echo "Type 1 for Sleeping-Bandit (AUER algorithm) crawler"
echo "Type 2 for Offline-DOM paths crawler (baseline)"
echo "Type 3 for Breadth-First Search crawler (baseline)"
echo "Type 4 for Depth-First Search crawler (baseline)"
echo "Type 5 for random crawler (baseline)"
read -p "Please enter the crawler's number you want to use: " method

case $method in
    1)
		clear
		echo "The use of Sleeping-Bandit crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Number of runs: " num_executions
		read -p "name of the directory in which to save crawling information (X in auer/logs/X): " log_path_all_sites
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		read -p "Similarity threshold for link-to-action mapping (float between 0 and 1): " threshold
		read -p "Parameter m in multiplicative hashing function: " m
		read -p "Parameter w in multiplicative hashing function: " w
		read -p "n in n-grams used in DOM path vector representation: " n
				read -p "Exploration--exploitation coefficient alpha (2s2 for 2sqrt2, float otherwise): " alpha
		read -p "Use URL classifier? 1 for Yes, anything else for No (then oracle is used): " use_url_classifier
		read -p "Training epoch size of URL classifier (if you answered \"Yes\" above): " batch_size_url
		echo ""
		echo "Launching $num_executions runs of Sleeping-Bandit crawler for site $site_name ..."
		for (( i=1; i<=$num_executions; i++ )); do
			python3 crawlers/auer/auer.py "$threshold" "$m" "$w" "$budget" "$n" "$batch_size_url" "$use_url_classifier" "$alpha" "$site_name" "$log_path_all_sites" &
			sleep 4
		done
	;;
    2)
		clear
		echo "The use of Offline-DOM paths crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "name of the directory in which to save crawling information (X in auer/logs/X): " log_path_all_sites
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		read -p "Similarity threshold for link-to-action mapping (float between 0 and 1): " threshold
		read -p "Parameter m in multiplicative hashing function: " m
		read -p "Parameter w in multiplicative hashing function: " w
		read -p "n in n-grams used in DOM path vector representation: " n
		echo ""
		echo "Launching a run of Offline-DOM paths crawler for site $site_name ..."
		python3 crawlers/offline_dom/offline_dom_crawler.py "$threshold" "$m" "$w" "$budget" "$n" 0 "$log_path_all_sites" "$site_name"
	;;
	3)
		clear
		echo "The use of Breadth-First Search crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "name of the directory in which to save crawling information (X in auer/logs/X): " log_path_all_sites
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of Breadth-First Search crawler for site $site_name ..."
		python3 crawlers/bfs/bfs.py "$budget" "$log_path_all_sites" "$site_name"
	;;
	4)
		clear
		echo "The use of Depth-First Search crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "name of the directory in which to save crawling information (X in auer/logs/X): " log_path_all_sites
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of Depth-First Search crawler for site $site_name ..."
		python3 crawlers/dfs/dfs.py "$budget" "$log_path_all_sites" "$site_name"
	;;
	5)

		clear
		echo "The use of random crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "name of the directory in which to save crawling information (X in auer/logs/X): " log_path_all_sites
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of random crawler for site $site_name ..."
		python3 crawlers/random/random_crawler.py "$budget" "$log_path_all_sites" "$site_name"
	;;
esac

wait
echo ""
echo "Local crawling run(s) finished."
