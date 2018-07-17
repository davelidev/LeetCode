class Answer(object):
'''274. H-Index'''
    def hIndex(citations):
        counter = [0] * (len(citations) + 1)
        for citation in citations:
            if citation > len(citations):
                citation = len(citations)
            counter[citation] += 1
        sum_from_right = 0
        for i in range(len(citations), -1, -1):
            sum_from_right += counter[i]
            if sum_from_right >= i:
                return i