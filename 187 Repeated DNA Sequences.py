class Answer(object):
'''187. Repeated DNA Sequences'''
    def findRepeatedDnaSequences(s):
        visited = set()
        res = []
        for i in range(0, len(s) - 10 + 1):
            sub_s = s[i: i + 10]
            if sub_s in visited and sub_s not in res:
                res.append(sub_s)
            visited.add(sub_s)
        return res