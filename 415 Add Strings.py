class Answer(object):
'''415. Add Strings'''
    def addStrings(num1, num2):
        def convert_to_int(num):
            res = 0
            for digit in num:
                res *= 10
                res += ord(digit) - ord('0')
            return res
        return str(convert_to_int(num1) + convert_to_int(num2))