class Answer(object):
'''575. Distribute Candies'''
    def distributeCandies(candies):
        return min(len(candies)/2, len(set(candies)))