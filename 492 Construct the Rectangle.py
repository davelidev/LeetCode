class Answer(object):
'''492. Construct the Rectangle'''
    def constructRectangle(area):
        i = j = int(area ** 0.5)
        while (i * j) != area:
            if i * j > area: i -= 1
            else: j += 1
        return j, i