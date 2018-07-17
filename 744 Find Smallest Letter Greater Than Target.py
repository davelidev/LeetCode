class Answer(object):
'''744. Find Smallest Letter Greater Than Target'''
    def nextGreatestLetter(letters, target):
        return next((char for char in letters if target < char), letters[0])