#!/bin/bash
clear
echo "Select the crawler you want to use for \"semi-online\" crawling (locally if already crawled, otherwise online):"
echo "Type 1 for Sleeping-Bandit (AUER algorithm) crawler"
echo "Type 2 for Focused Crawler (baseline)"
echo "Type 3 for Offline-DOM paths crawler (baseline)"
echo "Type 4 for Breadth-First Search crawler (baseline)"
echo "Type 5 for Depth-First Search crawler (baseline)"
echo "Type 6 for random crawler (baseline)"
read -p "Please enter the crawler's number you want to use: " method

current_dir=$(pwd)

case $method in
    1)
		clear
		echo "The use of Sleeping-Bandit crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		read -p "Similarity threshold for link-to-action mapping (float between 0 and 1): " threshold
		read -p "Parameter m in multiplicative hashing function: " m
		read -p "Parameter w in multiplicative hashing function: " w
		read -p "n in n-grams used in DOM path vector representation: " n
    read -p "Exploration--exploitation coefficient alpha (2s2 for 2sqrt2, float otherwise): " alpha
		echo ""
		echo "Launching a run of Sleeping-Bandit crawler for site $site_name ..."

    budget=${budget:--1}
    threshold=${threshold:-.75}
    m=${m:-12}
    w=${w:-15}
    n=${n:-2}
    alpha=${alpha:-2s2}

		project_path="$current_dir/crawlers/auer_crawler"

		PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="auer_crawler.settings" scrapy crawl auer_crawler \
    		-a site_name="$site_name" \
    		-a output_path="crawlers/auer_crawler/output" \
    		-a path_to_common_db="data" \
    		-a starting_url="$starting_url" \
    		-a alpha="$alpha" \
    		-a use_url_classifier="True" \
    		-a classifier_params="{'max_iter_optimizer': 100, 'random_state_optimizer': 42, 'n_in_ngrams': 2, 'batch_size': 10}" \
    		-a threshold="$threshold" \
    		-a m="$m" \
    		-a w="$w" \
    		-a n_grams_path_dom_path_representation="$n" \
    		-a budget="$budget"

	;;
    2)
    clear
    echo "The use of the Focused Crawler requires to set a certain number of parameters, which follows."
    read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
	read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
    read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
    echo ""
    echo "Launching a run of Focused Crawler for site $site_name ..."

    project_path="$current_dir/crawlers/focused_online_crawler"

	PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="focused_online_crawler.settings" scrapy crawl focused_online_crawler \
    	-a site_name="$site_name" \
    	-a output_path="crawlers/focused_online_crawler/output" \
    	-a path_to_common_db="data" \
    	-a starting_url="$starting_url" \
    	-a budget="$budget"
    ;;
    3)
		clear
		echo "The use of Offline-DOM paths crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		read -p "Similarity threshold for link-to-action mapping (float between 0 and 1): " threshold
		read -p "Parameter m in multiplicative hashing function: " m
		read -p "Parameter w in multiplicative hashing function: " w
		read -p "n in n-grams used in DOM path vector representation: " n
		echo ""
		echo "Launching a run of Offline-DOM paths crawler for site $site_name ..."

		project_path="$current_dir/crawlers/dom_off_online_crawler"

		PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="dom_off_online_crawler.settings" scrapy crawl dom_off_online_crawler \
		    -a site_name="$site_name" \
		    -a output_path="crawlers/dom_off_online_crawler/output" \
		    -a path_to_common_db="data" \
		    -a starting_url="$starting_url" \
		    -a threshold="$threshold" \
		    -a m="$m" \
		    -a w="$w" \
		    -a n_grams_path_dom_path_representation="$n" \
		    -a budget="$budget"

	;;
	4)
		clear
		echo "The use of Breadth-First Search crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of Breadth-First Search crawler for site $site_name ..."

		project_path="$current_dir/crawlers/bfs_online_crawler"

		PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="bfs_online_crawler.settings" scrapy crawl bfs_online_crawler \
		    -a site_name="$site_name" \
		    -a output_path="crawlers/bfs_online_crawler/output" \
		    -a path_to_common_db="data" \
		    -a starting_url="$starting_url" \
		    -a budget="$budget"
	;;
	5)
		clear
		echo "The use of Depth-First Search crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of Depth-First Search crawler for site $site_name ..."

		project_path="$current_dir/crawlers/dfs_online_crawler"

		PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="dfs_online_crawler.settings" scrapy crawl dfs_online_crawler \
		    -a site_name="$site_name" \
		    -a output_path="crawlers/dfs_online_crawler/output" \
		    -a path_to_common_db="data" \
		    -a starting_url="$starting_url" \
		    -a budget="$budget"
	;;
	6)
		clear
		echo "The use of random crawler requires to set a certain number of parameters, which follows."
		read -p "Name of the website's local replica you want to crawl (name of .db file without extension, X in data/X.db): " site_name
		read -p "Starting URL of the crawl (the homepage of the website, for instance): " starting_url
		read -p "Maximum number of crawling episodes (-1 for unlimited): " budget
		echo ""
		echo "Launching a run of random crawler for site $site_name ..."

		project_path="$current_dir/crawlers/random_online_crawler"

		PYTHONPATH="$project_path" SCRAPY_SETTINGS_MODULE="random_online_crawler.settings" scrapy crawl random_online_crawler \
		    -a site_name="$site_name" \
		    -a output_path="crawlers/random_online_crawler/output" \
		    -a path_to_common_db="data" \
		    -a starting_url="$starting_url" \
		    -a budget="$budget"
	;;
esac

wait
echo ""
echo "Local crawling run(s) finished."
