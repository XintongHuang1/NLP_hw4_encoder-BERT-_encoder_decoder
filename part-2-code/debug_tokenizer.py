from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# Test encoding
test_sql = "SELECT * FROM flight WHERE from_airport = 'bwi'"
test_nl = "translate to SQL for flight database: show me flights from baltimore"

# Encode
enc_ids = tokenizer.encode(test_nl, add_special_tokens=True)
tgt_ids = tokenizer.encode(test_sql, add_special_tokens=True)

print("Encoder input IDs:", enc_ids[:20])
print("Encoder text:", tokenizer.decode(enc_ids))
print()
print("Target IDs:", tgt_ids)
print("Target text:", tokenizer.decode(tgt_ids))
print()
print("Pad token ID:", tokenizer.pad_token_id)
print("EOS token ID:", tokenizer.eos_token_id)
print("Decoder start token:", tokenizer.pad_token_id)  # T5 uses pad as decoder start
print()

# Test what happens with decoder start
print("Should start decoder with:", tokenizer.pad_token_id)
