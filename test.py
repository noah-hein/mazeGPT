from transformers import PreTrainedTokenizerFast, pipeline, GPT2LMHeadModel

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../data/tokenizer.json")
    input_ids = tokenizer.encode("<start>")

    model = GPT2LMHeadModel.from_pretrained("../temp/checkpoint-800", local_files_only=True)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    test = generator("<start>", max_length=300)
    print(test)
