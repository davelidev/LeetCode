class Answer(object):
'''347. Top K Frequent Elements'''
    def topKFrequent(nums, k):
        from collections import Counter
        counter = Counter(nums)
        freq_to_val = {}
        for x in counter:
            freq_to_val.setdefault(counter[x], [])
            freq_to_val[counter[x]].append(x)
        keys = freq_to_val.keys()
        keys.sort(reverse=True)
        res = []
        for key in keys:
            val = freq_to_val[key]
            while val:
                res.append(val.pop())
                if len(res) == k:
                    return res
        return res