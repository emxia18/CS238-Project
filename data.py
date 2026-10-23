from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

articles = list(
    dataset.shuffle(seed=42, buffer_size=10_000).take(10_000)
)

print(articles[0]["title"])
print(articles[0]["text"][:500])