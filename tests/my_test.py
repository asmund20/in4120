from context import in3120


def simple_verify(finder, text, expected):
    results = list(finder.scan(text))
    print([(result.surface, result.match) for result in results], expected)


analyzer = in3120.SimpleAnalyzer()
trie = in3120.SimpleTrie.from_strings(["needle", "banana"], analyzer)

analyzer = in3120.Analyzer(in3120.SimpleNormalizer(), in3120.UnigramTokenizer())
finder = in3120.StringFinder(trie, analyzer)
expected_1 = ({"surface": "neEdle", "span": (8, 14), "match": "needle", "meta": None},)
expected_2 = ({"surface": "banana", "span": (53, 59), "match": "banana", "meta": None},)
text = "thereisaneEdleinthishaystacksomewhereiamsureotherwisebananapineapple"
results = list(finder.scan(text))

print(f"Expected results: {expected_1}\nand\n{expected_2}\n")
print(f"Actual results: {results}")
