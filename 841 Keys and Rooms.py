class Answer(object):'''841. Keys and Rooms'''
    def canVisitAllRooms(rooms):
        if not rooms: return True
        visited, bfs = {0}, {0}
        while bfs:
            bfs = {next_room
                   for room in bfs
                   for next_room in rooms[room]
                   if next_room not in visited and (visited.add(next_room) is None)}
        return len(visited) == len(rooms)