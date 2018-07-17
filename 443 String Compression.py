class Answer(object):
'''443. String Compression'''
    def compress(chars):
        i = 0
        chars.append('dummy')
        for j in range(len(chars)):
            if chars[i] == chars[j] and i != j:
                chars[j] = None
            elif  chars[i] != chars[j]:
                if j - i != 1:
                    count = str(j - i)
                    for k in range(len(count)): chars[i + k + 1] = count[k]
                i = j
        chars.pop()
        i = 0
        for j, char in enumerate(chars):
            if char is not None:
                chars[i] = char
                i += 1
        while len(chars) != i: chars.pop()
        return i
        
        434. Number of Segments in a String
    def countSegments(s):
        count = 0
        s = s.strip()
        prev = None
        for char in s:
            if prev == ' ' and char != ' ':
                count += 1
            prev = char
        if not count and s: return 1
        elif not count and not s: return 0
        elif count: return count + 1