import torch

from src.maxsim import maxsim_score


class ColBERTIndex:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.masks = []

    def build(self, model, documents):
        self.documents = list(documents)

        model.eval()

        with torch.no_grad():
            embeddings, masks = model.encode(documents)

        self.embeddings = [
            embeddings[i].cpu()
            for i in range(len(documents))
        ]

        self.masks = [
            masks[i].cpu()
            for i in range(len(documents))
        ]

    def search(self, model, query, top_k=5):
        model.eval()

        with torch.no_grad():
            query_embeddings, query_mask = model.encode([query])

        query_embeddings = query_embeddings[0].cpu()
        query_mask = query_mask[0].bool().cpu()

        query_embeddings = query_embeddings[query_mask]

        results = []

        for i, document in enumerate(self.documents):
            document_embeddings = self.embeddings[i]
            document_mask = self.masks[i].bool()

            document_embeddings = document_embeddings[document_mask]

            score = maxsim_score(
                query_embeddings,
                document_embeddings
            )

            results.append({
                "document": document,
                "score": score.item(),
                "index": i
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]

    def save(self, path):
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "masks": self.masks
        }

        torch.save(data, path)

    def load(self, path):
        data = torch.load(
            path,
            map_location="cpu"
        )

        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.masks = data["masks"]

    def __len__(self):
        return len(self.documents)