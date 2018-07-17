class Answer(object):
'''295. Find Median from Data Stream'''
    # maintain 2 heaps, one is min heap for storing the max half of the numbers, one is max heap for storing min half of the numbers. The maximun number is generated if push and poll from one of the heaps, then push into another.
    class MedianFinder(object):
        def __init__(self):
            self.larger = MinHeap()
            self.smaller = MaxHeap()
        def addNum(self, num):
            if len(self.smaller) == len(self.larger):
                self.smaller.push(self.larger.push(num).poll())
            else:
                self.larger.push(self.smaller.push(num).poll())
        def findMedian(self):
            if 0 < len(self.smaller) == len(self.larger):
                return (self.smaller.seek() + self.larger.seek())/2
            return self.smaller.seek()