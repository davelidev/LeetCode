class Answer(object):
'''406. Queue Reconstruction by Height'''
    def reconstructQueue(people):
        people.sort(key=lambda(h, k): (-h, k))
        res = []
        for p in people:
            res.insert(p[1], p)
        return res