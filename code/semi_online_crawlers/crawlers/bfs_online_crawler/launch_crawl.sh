#!/bin/bash

/home/gauquier/anaconda3/bin/python3 -m scrapy crawl bfs_online_crawler \
    -a site_name="nces2" \
    -a output_path="output" \
    -a path_to_common_db="../common_db" \
    -a starting_url="https://nces.ed.gov/" \
    -a budget="1000000.0"

