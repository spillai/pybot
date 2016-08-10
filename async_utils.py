#!/usr/bin/env python

# Blatantly stolen from http://code.activestate.com/recipes/576684-simple-threading-decorator/

import functools
import Queue
import threading
 
def run_async(func):
        """
        run_async(func)
        
        function decorator, intended to make "func" run in a separate
        thread (asynchronously).
        Returns the created Thread object
        
        E.g.:
        @run_async
        def task1():
          do_something

        @run_async
        def task2():
          do_something_too

        t1 = task1()
        t2 = task2()
        ...
        t1.join()
        t2.join()
	"""
	from threading import Thread
        from functools import wraps

	@wraps(func)
	def async_func(*args, **kwargs):
		func_hl = Thread(target=func, args=args, kwargs=kwargs)
		func_hl.start()
		return func_hl

	return async_func

def async_prefetch_wrapper(iterable, bufsize=100):
        """
	wraps an iterater such that it produces items in the background
	uses a bounded queue to limit memory consumption
	"""
	done = object()
	def worker(q,it):
		for item in it:
			q.put(item)
		q.put(done)

	# launch a thread to fetch the items in the background
	queue = Queue.Queue(bufsize)
	it = iter(iterable)
	thread = threading.Thread(target=worker, args=(queue, it))
	thread.daemon = True
	thread.start()

	# pull the items of the queue as requested
	while True:
		item = queue.get()
		# if item == done:
		# 	return
		# else:
                yield item
 
def async_prefetch(func):
	"""
	decorator to make generator functions fetch items in the background
	"""
	@functools.wraps(func)
	def wrapper(*args, **kwds):
		return async_prefetch_wrapper( func(*args, **kwds) )
	return wrapper

if __name__ == '__main__':
	from time import sleep

	@run_async
	def print_somedata():
		print 'starting print_somedata'
		sleep(2)
		print 'print_somedata: 2 sec passed'
		sleep(2)
		print 'print_somedata: 2 sec passed'
		sleep(2)
		print 'finished print_somedata'

	def main():
		print_somedata()
		print 'back in main'
		print_somedata()
		print 'back in main'
		print_somedata()
		print 'back in main'

	main()
