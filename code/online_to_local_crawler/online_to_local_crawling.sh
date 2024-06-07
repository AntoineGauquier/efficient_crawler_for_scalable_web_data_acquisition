#!/bin/bash

clear
echo "This script exhaustively crawls (using Bredth-First Search) a website to make a local replica (which can be used to conduct local crawming experiments later on)."
read -p "Enter the URL from which to start the crawl: " starting_url
read -p "Enter a name for the replicated website (the X in code/data/X.db): " site_name
read -p "Enter the maximum number of responses to crawl (an integer): " max_number_of_responses

python3 online_to_local_crawler/online_to_local_crawler/spiders/spider_online_to_local_crawler.py "$starting_url" "$site_name" "$max_number_of_responses"
