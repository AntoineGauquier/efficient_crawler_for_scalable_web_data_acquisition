#!/bin/bash

/home/gauquier/anaconda3/bin/python3 -m scrapy crawl auer_crawler \
    -a site_name="cnis" \
    -a output_path="output" \
    -a path_to_common_db="../common_db" \
    -a starting_url="https://www.cnis.fr/" \
    -a alpha="2.8284271247461903" \
    -a use_url_classifier="True" \
    -a classifier_params="{'max_iter_optimizer': 100, 'random_state_optimizer': 42, 'n_in_ngrams': 2, 'batch_size': 10}" \
    -a threshold="0.75" \
    -a m="12" \
    -a w="15" \
    -a n_grams_path_dom_path_representation="2" \
    -a budget="1000000.0"

