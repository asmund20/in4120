from context import in3120


def simple_verify(finder, text, expected):
    results = list(finder.scan(text))
    print([(result.surface, result.match) for result in results], expected)


analyzer = in3120.SimpleAnalyzer()
mesh = in3120.CorpusLoader.from_files(
    in3120.InMemoryCorpus(), ["../data/mesh.txt"]
)  # Contains more than 25K strings, including "medulla oblongata".
trie1 = in3120.SimpleTrie.from_strings(["medulla oblongata"], analyzer)
trie2 = in3120.SimpleTrie.from_strings((d["body"] or "" for d in mesh), analyzer)
finder1 = in3120.StringFinder(trie1, analyzer)
finder2 = in3120.StringFinder(trie2, analyzer)
buffer = "The injury was located close to the medulla oblongata."
results = list(finder1.scan(buffer))
print(
    f"Expected exactly one result, got: {len(results)}. The result there is: {results[0] if results else ''}"
)
