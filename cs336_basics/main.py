
class String:

    def __init__(self, name, age):
        self.name = name
        self.age = age


    def __repr__(self):
        return f"String({self.name}, {self.age})"

if __name__ == '__main__':

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    import regex as re

    a = re.findall(PAT, "I am a string~~! hda`")
    print(a)
