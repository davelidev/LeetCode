class Answer(object):
'''475. Heaters'''
    def findRadius(houses, heaters):
        heaters.sort()
        heaters.append(float('inf'))
        cur = diff = 0
        for house in sorted(houses):
            while cur + 1 < len(heaters) and heaters[cur + 1] < house: cur += 1
            diff = max(diff, min(abs(heaters[cur] - house), abs(heaters[cur + 1] - house)))
        return diff