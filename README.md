# Efficient Crawler for Scalable Web Data Acquisition

This repository provides additional content, code, and data supplementing the
research article *Efficient Crawler for Scalable Web Data Acquisition*.

## Additional content

[Supplementary material for the original paper](supplementary.pdf), with
detailed experimental results, additional related work, algorithms, and proof of
intractability of the graph crawling problem.

## Dataset

Since intellectual property forbids us to redistribute content we are not author of, we cannot provide direct access to the replicated websites used in the paper to conduct experiments. These datasets can be provided upon request, however. The original websites are available from the following URLs:

- <https://www.assemblee-nationale.fr/>
- <https://www.collectivites-locales.gouv.fr/>
- <https://www.cnis.fr/>
- <https://www.education.gouv.fr/>
- <https://www.ilo.org/>
- <https://www.interieur.gouv.fr/>
- <https://www.soumu.go.jp/>
- <https://www.justice.gouv.fr/>
- <https://www.psa.gov.qa/ar/Pages/default.aspx>

## Code

The source code of two types of crawlers are made available under the [MIT License](LICENSE).

### Local Crawlers

You will find in the [local_crawlers](code/local_crawlers/) folder, the code we used to conduct crawling experiments locally. Specifically:

* [data](code/local_crawlers/data/) is the place where locally replicated websites in the form of a `.db` file should be found.
* [crawlers](code/local_crawlers/crawlers/) contains the code of the different crawlers we present in the paper: our approach as well as baselines (one subdirectory for each crawler). [generic_local_crawler.py](code/local_crawlers/crawlers/generic_local_crawler.py) contains the local crawling framework used for all different crawlers, and [rl_actions.py](code/local_crawlers/crawlers/rl_actions.py) manages the actions when reinforcement learning is used.
* [crawl.sh](code/local_crawlers/crawl.sh) is a script that runs crawls by choosing a crawler, its parameters, etc.
* [graphical_results](code/local_crawlers/graphical_results/) contains the code used to generate plots from crawls runs, exactly as in the paper.
* [generate_plot.sh](code/local_crawlers/generate_plot.sh) is a script that generates plots by specifying where are the crawl results, which crawlers you want to integrate, etc.

### On-Line to Local Crawler

In [online_to_local_crawler](code/online_to_local_crawler), you will find a script [online_to_local_crawling.sh](code/online_to_local_crawler/online_to_local_crawling.sh) that can be used to make local replications of websites, to further conduct local crawling experiments, as presented above. The replications as stored are SQLite3 [^1] relational databases in the [data](code/local_crawlers/data/) folder. 

[^1]: https://docs.python.org/3/library/sqlite3.html
