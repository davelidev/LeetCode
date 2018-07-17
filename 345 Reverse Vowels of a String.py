class Answer(object):
'''345. Reverse Vowels of a String'''
    def reverseVowels(s):
        vowels = [char for char in s if char in 'aeiouAEIOU']
        return ''.join([ (char if char not in 'aeiouAEIOU' else vowels.pop()) for char in s])