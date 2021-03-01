import torch

__all__ = ["Distillation"]


class Distillation:
    """

    Parameters:
    -----------

        kb: Mkb model.

        tokenizer.add_tokens('universite paul sabatier')
        # Set the embedding of the new token as the mean of the previous tokens.
        # model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])
        model.resize_token_embeddings(len(tokenizer))

    """

    def __init__(self, kb, device):

        self.device = device

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

        self.cosine = {
            bert_id: cos(
                kb.entity_embedding[kb_id].detach().to(self.device),
                kb.entity_embedding.detach()[self.mask].to(self.device),
            )
            for bert_id, kb_id in tqdm.tqdm(
                self.kb.items(), desc="Cosine similarities", position=0
            )
        }
        pass

    def distill(self):
        pass
