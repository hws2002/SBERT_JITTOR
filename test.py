
import jittor as jt
from model.bert_model import BertConfig, BertModel
from objectve_functions.classification import SBERTClassificationHead


def main():
    config = BertConfig()
    encoder = BertModel(config)
    head = SBERTClassificationHead(hidden_size=config.hidden_size, num_labels=3)

    encoder.eval()
    head.eval()

    batch, seq_len = 2, 8
    dummy_ids = jt.randint(0, config.vocab_size, (batch, seq_len))

    outputs_a = encoder(dummy_ids, attention_mask=jt.ones((batch, seq_len)))
    outputs_b = encoder(dummy_ids, attention_mask=jt.ones((batch, seq_len)))

    emb_a = outputs_a["last_hidden_state"][:, 0, :]
    emb_b = outputs_b["last_hidden_state"][:, 0, :]

    logits = head(emb_a, emb_b)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    main()
