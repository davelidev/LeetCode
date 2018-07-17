class Answer(object):
'''332. Reconstruct Itinerary'''
    def findItinerary(tickets):
        from_to = collections.defaultdict(list)
        for dept, arr in tickets: from_to[dept].append(arr)
        for dept, arrs in from_to.iteritems(): arrs.sort(reverse=True)
        def dfs(airport, route=[]):
            while from_to[airport]:
                dfs(from_to[airport].pop(), route)
            route.append(airport)
            return route
        return dfs('JFK')[::-1]