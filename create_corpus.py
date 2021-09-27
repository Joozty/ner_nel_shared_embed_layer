import spacy
from spacy.tokens import DocBin
from spacy.kb import KnowledgeBase

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("sentencizer", first=True)

train_docs = DocBin()
test_docs = DocBin()

doc = nlp('Emerson Emerson')
doc.ents = [doc.char_span(0, 7, label="THING", kb_id="1"), doc.char_span(8, 15, label="THING", kb_id="1")]

kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=300)
kb.add_entity(entity="1", entity_vector=nlp("description").vector, freq=342)

kb.to_disk("kb")
nlp.to_disk("nlp")


train_docs.add(doc)
test_docs.add(doc)

train_docs.to_disk("training.spacy")
test_docs.to_disk("test.spacy")
