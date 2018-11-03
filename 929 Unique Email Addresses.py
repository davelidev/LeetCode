class Answer(object):'''929. Unique Email Addresses'''
    def numUniqueEmails(emails):
        res = set()
        for email in emails:
            email = email.split("@")
            email[0] = email[0].replace(".", "")
            email[0] = email[0][:email[0].find("+")]
            res.add(tuple(email))
        return len(res)