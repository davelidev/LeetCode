class Answer(object):
'''632. Smallest Range'''
    def smallest_range(lst_of_lsts):
        min_heap = MinHeap()
        greedy_iter = []
        for i, lst in enumerate(lst_of_lsts):
            if lst:
                min_heap.push([lst[i][0], [i, 0]])
                greedy_iter.append(lst[i][0])
            else:
                return
        shotest_dist = max(greedy_iter) - min(greedy_iter)
        while min_heap:
            val, coord = min_heap.pop()
            greedy_iter[coord[0]] = lst_of_lsts[coord[0]][coord[1]]
            shotest_dist = max(greedy_iter) - min(greedy_iter)
            if coord[1] + 1 >= len(lst_of_lsts[coord[0]]):
                return shotest_dist
            min_heap.push([lst[coord[0]][coord[1] + 1], [coord[0], coord[1] + 1]])