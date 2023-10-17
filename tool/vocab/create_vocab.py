vocab_words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "C", "G", "U"]
unused_tokens_count = 100

with open("G:\Challenges\RNA\code\SR-RFC\data\\vocab2.txt", "w") as vocab_file:
    unused_counter = 0
    for word in vocab_words:
        vocab_file.write(word + "\n")

        for i in range(unused_tokens_count):
            vocab_file.write("[unused" + str(unused_counter) + "]\n")
            unused_counter += 1

print("Vocabulary file 'vocab.txt' has been created.")
