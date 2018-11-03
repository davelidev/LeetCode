class Answer(object):'''49. Group Anagrams'''
    def groupAnagrams(strs):
        group = {}
        for word in strs:
            hashed_bucket = [0] * 26
            for char in word:
                hashed_bucket[ord(char) % len(hashed_bucket)] += 1
            key = hash(str(hashed_bucket))
            group.setdefault(key, [])
            group[key].append(word)
        return group.values()