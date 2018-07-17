class Answer(object):
'''480. Sliding Window Median'''
    # a variation of "295. Find Median from Data Stream", with 1 additional function, remove the a specific element from the heap(can be done using hashmap). At each sliding window, remove the element on the left and add the element on the right.