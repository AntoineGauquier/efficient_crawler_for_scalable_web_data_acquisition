#!/bin/bash

/home/gauquier/anaconda3/bin/python -m scrapy crawl dom_off_online_crawler \
    -a site_name="cnis" \
    -a output_path="output" \
    -a path_to_common_db="../common_db" \
    -a starting_url="https://www.cnis.fr/" \
    -a threshold="0.75" \
    -a m="12" \
    -a w="15" \
    -a n_grams_path_dom_path_representation="2" \
    -a budget="1000000.0"

