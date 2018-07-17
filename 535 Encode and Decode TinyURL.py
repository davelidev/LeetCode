class Answer(object):
'''535. Encode and Decode TinyURL'''
    class Codec:
        
        def __init__(self):
            self.num_to_url = []
        
        def encode(self, longUrl):
            self.num_to_url.append(longUrl)
            return "http://tinyurl.com/" + str(len(self.num_to_url))

        def decode(self, shortUrl):
            return self.num_to_url[int(shortUrl[int(shortUrl.rfind('/')) + 1:]) - 1]