class Answer(object):
'''599. Minimum Index Sum of Two Lists'''
    def findRestaurant(list1, list2):
        rest_to_idx_2 = {rest: i for i, rest in enumerate(list2)}
        min_dist, min_rests = float('inf'), []
        for i, rest in enumerate(list1):
            if rest in rest_to_idx_2:
                fav_sum = rest_to_idx_2[rest] + i
                if fav_sum < min_dist: min_dist, min_rests = fav_sum, [rest]
                elif fav_sum == min_dist: min_rests.append(rest)
        return min_rests