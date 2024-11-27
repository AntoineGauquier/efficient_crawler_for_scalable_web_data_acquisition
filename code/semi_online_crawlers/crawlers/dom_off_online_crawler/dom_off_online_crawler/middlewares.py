# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter

from scrapy.exceptions import IgnoreRequest, CloseSpider, StopDownload
#from scrapy.downloadermiddlewares.http11 import ResponseNeverReceived
from twisted.internet.error import TimeoutError, DNSLookupError, ConnectionRefusedError
from twisted.internet.defer import CancelledError, Deferred
from scrapy.http import Request, Response, TextResponse
import ast

class DomOffOnlineCrawlerSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class DomOffOnlineDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class CustomErrorAndRequestHandlingMiddleware:
    def __init__(self, crawler):
        self.crawler = crawler
        self.response_to_forward = None
        crawler.signals.connect(self.headers_received, signal=signals.headers_received)

    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware

    def spider_opened(self, spider):
        self.spider = spider

    def process_request(self, request, spider):
        cursor = spider.connection_common_db.cursor()
        query_str = "SELECT http_response, headers, body, content_length FROM {} WHERE url=?".format(spider.name_of_table)
        query = cursor.execute(query_str, (request.url,))
        result = query.fetchall()

        if len(result) == 0:
            return None

        assert(len(result) == 1)

        spider.logger.info("Webpage at URL \"" + request.url + "\" was found in common_db.")
        result = result[0]
        http_response = result[0]
        headers = result[1]
        body = result[2]
        content_length = result[3]

        response = TextResponse(
            url=request.url,
            status=int(http_response),
            body=body.encode('utf-8'),
            encoding='utf-8',
            headers=ast.literal_eval(headers),
            request=request)

        response.meta['from_common_db'] = True
        response.meta['content_length'] = int(content_length)

        return response

    def headers_received(self, headers, body_length, request, spider):
        content_type = headers.get(b'Content-Type', b'').decode('utf-8')
        content_length = headers.get(b'Content-Length', None)

        if content_length is not None and content_type and "html" not in content_type and headers.get(b'Location', None) is None:
            self.response_to_forward = TextResponse(url=request.url, headers=headers, status=200, body = b'', request=request)
            raise StopDownload(fail=False)

    def process_response(self, request, response, spider):
        if self.response_to_forward is not None:
            response_to_return = self.response_to_forward
            self.response_to_forward = None
            spider.logger.info("Truncated body Iteration for URL \"" + response.url + "\".")
            return response_to_return

        return response

    def process_exception(self, request, exception, spider):
        spider.logger.info(f"Error processing request {request.url}: {exception} (caught in middleware)")

        spider.current_number_of_requests_in_scrapy_queue -= 1 # If we get an error, we do not go through parse and do not update the number of requests in scrapy's queue.

        if spider.nb_episodes > 0:
            link = request.meta['link']

            if link.supposed_to_be_html:
                    spider.update_index_resource_supposed_to_be_html_but_is_not(link)

        request_to_add = spider.update_scrapy_queue_if_needed(is_middleware=True)
        
        if request_to_add is not None:
            spider.current_number_of_requests_in_scrapy_queue += 1
            spider.yielded_to_scrapy.add(request_to_add.meta['link'])
            spider.crawler.engine.crawl(request_to_add)#, spider)
       
        return None
