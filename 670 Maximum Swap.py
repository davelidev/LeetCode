class Answer(object):
'''670. Maximum Swap'''
    def maximumSwap(num):
        num = [int(i) for i in list(str(num))]
        max_from_right = [None] * len(num)
        for i in range(len(num) - 1, -1, -1):
            if i == len(num) - 1 or num[max_from_right[i + 1]] < num[i]:
                max_from_right[i] = i
            else: max_from_right[i] = max_from_right[i + 1]
        for i in range(len(num)):
            if max_from_right[i] != i and num[i] != num[max_from_right[i]] :
                num[i], num[max_from_right[i]] = num[max_from_right[i]], num[i]
                break
        return int(''.join([str(item) for item in num]))