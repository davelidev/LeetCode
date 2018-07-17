class Answer(object):
'''771. Jewels and Stones'''
    def numJewelsInStones(J, S):
        jewel_set = set(list(J))
        return sum(stone in jewel_set for stone in S)