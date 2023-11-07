from evaluate import fast_eval_ppl_wiki2
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_fast_eval_ppl_wiki2():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    fast_eval_ppl_wiki2(model, tokenizer, 1, 100)


if __name__ == "__main__":
    test_fast_eval_ppl_wiki2()
