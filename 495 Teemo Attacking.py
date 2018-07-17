class Answer(object):
'''495. Teemo Attacking'''
    def findPoisonedDuration(timeSeries, duration):
        total_time = 0
        for i, time in enumerate(timeSeries):
            total_time += duration
            time_diff = timeSeries[i] - timeSeries[i-1]
            if i > 0 and time_diff < duration:
                total_time -= duration - time_diff
        return total_time