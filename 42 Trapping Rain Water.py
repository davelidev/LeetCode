class Answer(object):
'''42. Trapping Rain Water'''
    #Have 2 pointers, one from the left, one from the right. Keep track of the max from the left as well as the max from the right. The water level in the current block is calculated by min(left_max, right_max) - cur_ground. move i from the left to right if max_left < max_right, move j fromt he right to left  otherwise.