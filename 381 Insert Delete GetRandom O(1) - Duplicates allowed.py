class Answer(object):
'''381. Insert Delete GetRandom O(1) - Duplicates allowed'''
    # need to edit the bellow code to allow for duplicates, assume no collision
    from random import randint
    class RandomizedCollection(object):
        def __init__(self, size=1028):
            self.size = size
            self.idx_vals_pair = [None] * size
            self.idx_mapper = []

        def hash_algo(self, val):
            return hash(val)  # assume there's more sophisticated method to do so

        def insert(self, val):
            self.idx_vals_pair[self.hash_algo(val) % self.size] = [len(self.idx_mapper), val]
            self.idx_mapper.append(self.hash_algo(val) % self.size)

        def remove(self, val):
            idx = self.idx_vals_pair[self.hash_algo(val) % self.size][0]
            self.idx_vals_pair[self.hash_algo(val) % self.size] = None
            self.idx_mapper[idx] = None
            if len(self.idx_mapper) == 1:
                self.idx_mapper.pop()
            else:
                new_idx = self.idx_mapper.pop()
                if not new_idx:
                    return
                self.idx_vals_pair[new_idx][0] = idx
                self.idx_mapper[idx] = new_idx

        def getRandom(self):
            if not self.idx_mapper:
                return None
            return self.idx_vals_pair[self.idx_mapper[randint(0, len(self.idx_mapper) - 1)]][1]