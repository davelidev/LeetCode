class Answer(object):
'''717. 1-bit and 2-bit Characters'''
    def isOneBitCharacter(bits):
        i = 0
        is_one_bit = False
        while i < len(bits):
            is_one_bit = not bits[i]
            if not bits[i]: i += 1
            else: i += 2
        return is_one_bit

    def isOneBitCharacter(bits):
        import re
        return re.findall('(1.|0)', reduce(lambda x, y : x + str(y), bits, ''))[-1] == '0'