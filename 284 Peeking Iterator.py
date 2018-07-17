class Answer(object):
'''284. Peeking Iterator'''
    class PeekingIterator(object):
        def __init__(self, iterator):
            self.iterator = iterator
            self.cache = None

        def peek(self):
            if self.cache is None and self.iterator.hasNext():
                self.cache = self.iterator.next()
            return self.cache

        def next(self):
            self.peek()
            cache = self.cache
            self.cache = None
            return cache

        def hasNext(self):            self.peek()
            return not bool(self.cache is None)