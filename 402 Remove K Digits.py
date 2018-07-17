class Answer(object):
'''402. Remove K Digits'''
    num = [int(x) for x in list(num)]
    i = 1
    while i < len(num) and k != 0:
        if num[i] > num[i - 1]:
            num.pop(i - 1)
            k -= 1
        else:
            i += 1
    num = ''.join([str(dig) for dig in num])