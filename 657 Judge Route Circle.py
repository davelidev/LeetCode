class Answer(object):
'''657. Judge Route Circle'''
    def judgeCircle(moves):
        from collections import Counter
        counts = Counter(moves)
        return counts.get('U', 0) == counts.get('D', 0) and counts.get('L', 0) == counts.get('R', 0)