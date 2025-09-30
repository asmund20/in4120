from context import in3120


analyzer = in3120.SimpleAnalyzer()
corpus1 = in3120.CorpusLoader.from_files(in3120.InMemoryCorpus(), ["../data/mesh.txt"])
corpus2 = in3120.AccessLoggedCorpus(corpus1)
inner = in3120.InMemoryInvertedIndex(corpus1, ["body"], analyzer)
index = in3120.AccessLoggedInvertedIndex(inner)
engine = in3120.SimpleSearchEngine(corpus2, index)
ranker = in3120.SimpleRanker()
query = "Water  polluTION"
options = in3120.SimpleSearchEngine.Options(match_threshold=0.5, hit_count=1)
matches = list(engine.evaluate(query, ranker, options))
assert matches is not None
history = corpus2.get_history()
print(f"History: {history}\nShould be: [25274], only the document in the result should be accessed.")
print(f"Matches: {matches}")
ordering1 = [('water', 3078),  # Document-at-a-time ordering if evaluated as "water pollution".
             ('pollution', 788), ('pollution', 789), ('pollution', 790), ('pollution', 8079),
             ('water', 8635),
             ('pollution', 23837),
             ('water', 9379), ('water', 23234), ('water', 25265),
             ('pollution', 25274),
             ('water', 25266), ('water', 25267), ('water', 25268), ('water', 25269), ('water', 25270),
             ('water', 25271), ('water', 25272), ('water', 25273), ('water', 25274), ('water', 25275),
             ('pollution', 25275),
             ('water', 25276),
             ('pollution', 25276),
             ('water', 25277), ('water', 25278), ('water', 25279), ('water', 25280), ('water', 25281)]
ordering2 = [('pollution', 788),  # Document-at-a-time ordering if evaluated as "pollution water".
             ('water', 3078),
             ('pollution', 789), ('pollution', 790), ('pollution', 8079),
             ('water', 8635),
             ('pollution', 23837),
             ('water', 9379), ('water', 23234), ('water', 25265),
             ('pollution', 25274),
             ('water', 25266), ('water', 25267), ('water', 25268), ('water', 25269), ('water', 25270),
             ('water', 25271), ('water', 25272), ('water', 25273), ('water', 25274),
             ('pollution', 25275),
             ('water', 25275),
             ('pollution', 25276),
             ('water', 25276), ('water', 25277), ('water', 25278), ('water', 25279), ('water', 25280),
             ('water', 25281)]
history = index.get_history()
test1 = history in (ordering1, ordering2)  # Strict. Preferred.
test2 = (len(history) == len(ordering1)) and (set(history) == set(ordering1))
if test2 and not test1:
    print("Unorthodox traversal ordering detected. This may or may not be benign.")
assert test1 or test2
