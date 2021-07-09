"""Config
"""
class Config:
    number_of_retries = 3
    seconds_between_retries = 2
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)
