class Answer(object):
'''630. Course Schedule III'''
    def scheduleCourse(courses):
        import heapq
        courses = sorted([(deadline, dur) for dur, deadline in courses])
        max_heap = []
        max_len = total_dur = 0
        for deadline, dur in courses:
            total_dur += dur
            heapq.heappush(max_heap, -dur)
            while total_dur > deadline:
                total_dur += heapq.heappop(max_heap)
            max_len = max(max_len, len(max_heap))
        return max_len