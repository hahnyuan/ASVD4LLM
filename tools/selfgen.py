import torch
from datasets import load_dataset


def gen_and_save(model, tokenizer, model_id, device, n_calib_samples=32, seqlen=2048):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    file_name = f"output/selfgen_{model_id.replace('/','_')}.txt"
    traindataset = []
    with torch.no_grad():
        for i in range(n_calib_samples):
            print(f"== {i} ==")
            text = traindata[i]["text"]
            new_text = text.strip()[:50]
            tokens = tokenizer(new_text, return_tensors="pt", max_length=seqlen, truncation=True)
            # print(tokens.input_ids[0,:2])
            if tokens.input_ids.shape[1] < 2:
                continue
            input_ids = tokens.input_ids[:, :2].to(device)
            all_input_ids = input_ids
            past_key_values = None
            for i in range(seqlen):
                out = model.forward(input_ids, past_key_values=past_key_values)
                p = torch.softmax(out.logits[0, -1], dim=-1)
                sample_index = torch.multinomial(p, num_samples=1)
                sample_token_id = sample_index.item()
                sample_token = tokenizer.decode(sample_token_id)
                input_ids = sample_index.unsqueeze(0)
                all_input_ids = torch.cat([all_input_ids, input_ids], dim=1)
                past_key_values = out.past_key_values
                if i % 100 == 0:
                    print(tokenizer.decode(all_input_ids[0]))
            traindataset.append({"input_ids": all_input_ids, "attention_mask": torch.ones_like(all_input_ids)})
            with open(file_name, "a") as f:
                f.write(tokenizer.decode(all_input_ids[0]) + "\n\n")
        torch.save(traindataset, f"cache/selfgen_{model_id.replace('/','_')}_{n_calib_samples}_{2048}_{42}.pt")
