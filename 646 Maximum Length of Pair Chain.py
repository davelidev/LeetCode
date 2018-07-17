class Answer(object):
'''646. Maximum Length of Pair Chain'''
    def findLongestChain(pairs):
        pairs.sort()
        cur = None
        count = 0
        for itv in reversed(pairs):
            if cur == None or cur[0] > itv[1]:
                cur = itv
                count += 1
        return count