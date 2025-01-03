import nltk


class WordAugTokenizeWrapper:
    def __init__(self, word_type='noun'):
        assert word_type in ['noun', 'noun_phrase']
        self.word_type = word_type

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(
                    ' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def remove_not_noun(self, nouns):
        not_noun_list = list("abcdefghijklmnopqrstuvwxyz<>")
        for word in not_noun_list:
            if word in nouns:
                nouns.remove(word)
        return nouns

    def __call__(self, text):
        # tags = ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ']
        tags = ['NN', 'NNP', 'NNS', 'NNPS']
        assert isinstance(text, str)
        tokenized = nltk.word_tokenize(text)
        nouns = []
        if len(tokenized) > 0:
            if self.word_type == 'noun':
                nouns = self.get_tag(tokenized, tags)
            elif self.word_type == 'noun_phrase':
                nouns = self.get_noun_phrase(tokenized)
            else:
                raise ValueError('word_type must be noun or noun_phrase')

        nouns = set(nouns)
        nouns = self.remove_not_noun(nouns)
        return list(nouns)
