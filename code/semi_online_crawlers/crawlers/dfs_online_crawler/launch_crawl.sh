#!/bin/bash

/home/gauquier/anaconda3/bin/python3 -m scrapy crawl dfs_online_crawler \
    -a site_name="cnis" \
    -a output_path="output" \
    -a path_to_common_db="../common_db" \
    -a starting_url="https://www.cnis.fr/" \
    -a budget="1000000.0"

