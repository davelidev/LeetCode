class Answer(object):
'''690. Employee Importance'''
    def getImportance(employees, id):
        id_to_person = {person.id: person for person in employees}
        adjs = {person: [id_to_person[adj_id] for adj_id in person.subordinates] for person in employees}
        visited = set()
        bfs = [id_to_person[id]]
        importance = 0
        while bfs:
            importance += sum(person.importance for person in bfs)
            bfs = [adj for person in bfs for adj in adjs[person] if adj not in visited and (not visited.add(adj))]
        return importance