class Answer(object):
'''223. Rectangle Area'''
    def computeArea(A, B, C, D, E, F, G, H):
        width, height = (min(C, G) - max(A, E)), (min(D, H) - max(B, F))
        overlap = width * height if width > 0 and height > 0 else 0
        area1 = (C - A) * (D - B)
        area2 = (G - E) * (H - F)
        return area1 + area2 - overlap