class Answer(object):
'''151. Reverse Words in a String'''
    " ".join(filter(lambda x: x != "", s.split(" "))[::-1])