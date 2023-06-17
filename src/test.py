from transformers import PreTrainedTokenizerFast, BertForMaskedLM, pipeline

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../data/tokenizer.json")
    input_ids = tokenizer.encode("<start>")


    model = BertForMaskedLM.from_pretrained("../temp", "checkpoint-800", local_files_only=True)

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)