class Answer(object):
'''383. Ransom Note'''
    def canConstruct(ransomNote, magazine):
        counts = [0] * 26
        for char in magazine: counts[ord(char) % len(counts)] += 1
        for char in ransomNote: counts[ord(char) % len(counts)] -= 1
        return all(count >= 0 for count in counts)