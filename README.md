# Efficient Crawler for Scalable Web Data Acquisition

This repository provides additional content, code, and data supplementing the
research article *Efficient Crawler for Scalable Web Data Acquisition* by Antoine Gauquier,
[Ioana Manolescu](https://pages.saclay.inria.fr/ioana.manolescu/), and [Pierre Senellart](https://pierre.senellart.com/).

## Additional content

[Supplementary material for the original paper](supplementary.pdf), with
detailed experimental results, additional related work, and proof of
intractability of the graph crawling problem.

## Dataset

Since intellectual property forbids us to redistribute content we are not author of, we cannot provide direct access to the replicated websites used in the paper to conduct experiments. We are therefore working on making this data available through a third party who takse responsibility for managing the rights associated with redistribution.

## Code

The source code of two types of crawlers are made available.

### Local Crawlers

You will find in the [local\_crawlers](code/local_crawlers/) folder, the code we used to conduct crawling experiments locally. More especially:
* [data](code/local_crawlers/data/) is the place where locally replicated websites in a form of a `.db` file should be found.
* [crawlers](code/local_crawlers/crawlers/) contains the code of the different crawlers we present in the paper: our approach as well as baselines (one subdirectory for each crawler). [generic\_local\_crawler.py](code/local_crawlers/crawlers/generic_local_crawler.py) contains the local crawling framework used for all different crawlers, and [rl\_actions.py](code/local_crawlers/crawlers/rl_actions.py) manages the actions when reinforcement learning is used.
* [crawl.sh](code/local_crawlers/script.sh) is a script that runs crawls by choosing a crawler, its parameters, etc.
* [graphical\_results](code/graphical_results/) contains the code used to generate plots from crawls runs, exactly as in the paper. 
* [generate\_plots.sh](code/graphical_results/generate_plots.sh) is a script that generates plots by specifying where are the crawl results, which crawlers you want to integrate, etc.

Once the replicated websites we used in the paper will be made available, all of this will make possible the re-generation of the results presented in the paper.

### On-Line Crawler

[online\_crawler](code/onlince_crawler) is not available yet, but will provide the efficient crawler we present in an on-line fashion, directly crawling websites by doing real HTTP queries over Web servers. This crawler *will not* allow the replication of the results presented in the paper, as a website (and therefore, its content), always dynamically evolves over time.

## Contact

<https://github.com/AntoineGauquier/efficient_crawler_for_scalable_web_data_acquisition/>

* Antoine Gauquier <antoine.gauquier@ens.psl.eu>
* Ioana Manolescu <ioana.manolescu@inria.fr>
* Pierre Senellart <pierre@senellart.com>
