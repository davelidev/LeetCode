class Answer(object):
'''556. Next Greater Element III'''
    def nextGreaterElement(n):
        num = list(str(n))
        n = len(num)
        for i in range(n - 2, -1, -1):
            if num[i] < num[i + 1]:
                next_largest = i + 1
                for j in range(i + 2, n):
                    if num[i] < num[j] <= num[next_largest]:
                        next_largest = j
                num[i], num[next_largest] = num[next_largest], num[i]
                next_greater = int(''.join(num[:i + 1] + num[i + 1:][::-1]))
                return next_greater if next_greater < (1<<31) else -1
        return -1