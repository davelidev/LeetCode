class Answer(object):
'''225. Implement Stack using Queues'''
    class MyStack(object):
        def __init__(self):
            import Queue
            self.pri, self.sec = Queue.PriorityQueue(), Queue.PriorityQueue()

        def push(self, x):
            self.pri.put(x)

        def pop(self):
            while True:
                tmp = self.pri.get() if not self.pri.empty() else None
                if not self.pri.empty(): self.sec.put(tmp)
                else:
                    self.pri, self.sec = self.sec, self.pri
                    return tmp

        def top(self):
            while True:
                tmp = self.pri.get() if not self.pri.empty() else None
                if tmp is not None: self.sec.put(tmp)
                if self.pri.empty():
                    self.pri, self.sec = self.sec, self.pri
                    return tmp

        def empty(self):
            return self.pri.empty()