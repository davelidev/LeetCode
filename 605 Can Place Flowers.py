class Answer(object):
'''605. Can Place Flowers'''
    def canPlaceFlowers(flowerbed, n):
        count = 0
        for i, planted in enumerate(flowerbed):
            if not planted:
                if (i == 0 or not flowerbed[i-1]) and (i == (len(flowerbed) - 1) or not flowerbed[i+1]):
                    flowerbed[i] = 1
                    count += 1
            if count >= n: return True
        return False