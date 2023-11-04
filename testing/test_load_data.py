from datautils import sample_train_loaders
from transformers import GPT2Tokenizer


def test_c4_sample_train_loaders():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    trainloader = sample_train_loaders(
        "c4",
        tokenizer,
    )
    print(trainloader)


if __name__ == "__main__":
    test_c4_sample_train_loaders()
