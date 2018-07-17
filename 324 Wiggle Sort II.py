class Answer(object):
'''324. Wiggle Sort II'''
    # Iterate through each number from left to right of a b c d e f. If a > b, then it's out of order, since b is in even position, and b is less than the previous number a, then swap a and b inplace, otherwise continue. Do the same for even position elements.
    # O(n) time, O(1) space