class Answer(object):
'''401. Binary Watch'''
    def readBinaryWatch(num):
        def count_ones(num):
            count = 0
            while num != 0:
                count += num % 2
                num /= 2
            return count
        return [
            "%d:%02d" %(hr, m)
            for hr in range(12)
            for m in range(60)
            if count_ones(hr) + count_ones(m) == num
        ]