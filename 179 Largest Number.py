class Answer(object):
'''179. Largest Number'''
    def largestNumber(nums):
        comp_func = lambda x, y : 1 if str(x) + str(y) > str(y) + str(x) else -1
        return str(int(''.join(sorted(map(lambda x: str(x), nums), reverse=True, cmp=comp_func))))