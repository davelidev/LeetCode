class Answer(object):'''911. Online Election'''
import bisect
class TopVotedCandidate(object):

    def __init__(self, persons, times):        
        person_to_count = {}
        max_vote = 0
        winning = self.winning = []
        self.times = times
        for i in range(len(persons)):
            person_to_count[persons[i]] = person_to_count.get(persons[i], 0) + 1
            if person_to_count[persons[i]] >= max_vote:
                max_vote = person_to_count[persons[i]]
                winning.append(persons[i])
            else:
                winning.append(winning[-1])            

    def q(self, t):
        idx = bisect.bisect_left(self.times, t)
        if idx >= len(self.times) or self.times[idx] > t: idx -= 1
        return self.winning[idx]