from context import in3120

analyzer = in3120.SimpleAnalyzer()
corpus = in3120.InMemoryCorpus()

corpus.add_document(in3120.InMemoryDocument(0, {"field1": "a b c", "field2": "b c d"}))
corpus.add_document(in3120.InMemoryDocument(1, {"field1": "x", "field2": "y"}))
corpus.add_document(in3120.InMemoryDocument(2, {"field1": "y", "field2": "z"}))
# engine1 = in3120.SuffixArray(corpus, ["field1"], analyzer)
# engine2 = in3120.SuffixArray(corpus, ["field2"], analyzer)

print(f"Corpus: {[doc for doc in iter(corpus)]}")

engine0 = in3120.SuffixArray(corpus, ["field1", "field2"], analyzer)

print(f"Haystack: {engine0._haystack}")
print(f"Suffixes: {engine0._suffixes}")


while True:
    query = input("Query>")
    print(list(engine0.evaluate(query)))

print(list(engine0.evaluate("a b c b")))
print()
print(list(engine0.evaluate("b")))
print()
print(list(engine0.evaluate("y")))
