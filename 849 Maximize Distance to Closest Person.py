class Answer(object):'''849. Maximize Distance to Closest Person'''
    def maxDistToClosest(seats):
        max_dist = prev = 0
        for i in range(1, len(seats)):
            if not seats[i - 1] and seats[i]:
                max_dist = max(max_dist, i - prev)
            elif seats[i - 1] and not seats[i]:
                prev = i - 1
        return max(max_dist / 2,  # max gap
                   next((i for i, seated in enumerate(seats) if seated == 1)),  # from left
                   next((i for i in range(len(seats)) if seats[len(seats) - i - 1]))  # from right
                  )