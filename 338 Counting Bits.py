class Answer(object):
'''338. Counting Bits'''
    def countBits(self, num):
        res = [0]
        for i in range(1, num + 1):
            res.append(res[i >> 1] + (i & 1))
        return res