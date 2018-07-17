class Answer(object):
'''557. Reverse Words in a String III'''
    def reverseWords(s): return ' '.join([word[::-1] for word in s.split(' ')])