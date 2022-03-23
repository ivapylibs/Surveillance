ACT_CODEBOOK = {
    "i": "Pick",
    "l": "Place"
}

class ActDecoder():
    no_act_label = "No Activity"
    def __init__(self) -> None:
        self.key_cache = None
        self.activity = ActDecoder.no_act_label
        
        self.act_codebook = ACT_CODEBOOK

    def decode(self, key):
        # if no activity (then shouldn't be any key cache)
        if self.activity == ActDecoder.no_act_label:
            self.activity = self.act_codebook[key]
            self.key_cache = key
        else:
            assert key == self.key_cache, "The key does not match. New key: \'{}\'. Old key: \'{}\'".format(key, self.key_cache)
            self.activity = ActDecoder.no_act_label
            self.key_cache = None

    def get_activity(self):
        return self.activity