class Answer(object):
'''232. Implement Queue using Stacks'''
    class MyQueue(object):

        def __init__(self):
            self.incoming = []
            self.outgoing = []

        def push(self, x): self.incoming.append(x)
        
        def _move_to_outgoing(self):
            if not self.outgoing:
                while self.incoming:
                    self.outgoing.append(self.incoming.pop())
            return self.outgoing

        def pop(self): return self._move_to_outgoing().pop()

        def peek(self): return self._move_to_outgoing()[-1]

        def empty(self): return not (self.incoming or self.outgoing)