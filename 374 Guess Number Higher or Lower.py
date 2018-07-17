class Answer(object):
'''374. Guess Number Higher or Lower'''
    def guessNumber(n):
        low, high = 1, n
        while low <= high:
            mid = (low + high) / 2
            print low, high, mid
            if guess(mid) == 0: return mid
            elif guess(mid) > 0: low = mid + 1
            else: high = mid - 1