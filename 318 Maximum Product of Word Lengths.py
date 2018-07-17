class Answer(object):
'''318. Maximum Product of Word Lengths'''
    def maxProduct(words):
        bit_wise_words = []
        for word in words:
            int_word = 0
            for char in word:
                int_word |= 1 << ord(char) % 26
            bit_wise_words.append(int_word)

        max_length = 0
        for i in range(len(words)):
            for j in range(i, len(words)):
                length = len(words[i]) * len(words[j])
                if bit_wise_words[i] & bit_wise_words[j] == 0 and max_length < length:
                    max_length = length
        return max_length