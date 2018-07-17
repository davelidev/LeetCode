class Answer(object):
'''729. My Calendar I'''
    class MyCalendar(object):

        def __init__(self):
            self.intervals = []

        def book(self, start, end):
            for int_start, int_end in self.intervals:
                if not (int_end <= start or end <= int_start):
                    return False
            self.intervals.append((start,end))
            return True