class Answer(object):'''3. Longest Substring Without Repeating Characters'''
    keep a set of visited chars, iterate through the string. Keep count of number of visited chars, update it in each iteration. If the char is already visited, remove from the set and reset the count.
    O(n) time, O(n) space