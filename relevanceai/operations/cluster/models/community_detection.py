import numpy as np

from relevanceai.constants import MissingPackageError


class CommunityDetection:
    def __init__(
        self,
        threshold=0.75,
        min_community_size=10,
        init_max_size=1000,
        gpu=False,
    ):

        self.gpu = gpu
        self.threshold = threshold
        self.min_community_size = min_community_size
        self.init_max_size = init_max_size

    def __call__(self, *args, **kwargs):
        return self.fit_predict(*args, **kwargs)

    def fit_predict(self, vectors):
        if self.gpu:
            communities = self.community_detection_gpu(vectors)
        else:
            communities = self.community_detection_cpu(vectors)

        labels = [-1 for _ in range(vectors.shape[0])]
        for cluster_index, community in enumerate(communities):
            for index in community:
                labels[index] = cluster_index

        return np.array(labels)

    def cosine(self, embeddings):
        """
        effecient cosine sim
        """
        similarity = np.dot(embeddings, embeddings.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        return cosine

    def topk(self, embeddings, k):
        """
        numpy topk
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        indices = embeddings.argpartition(-k, axis=1)[:, -k:]
        indices = np.flip(indices, 1)

        values = np.flip(
            np.sort(embeddings[np.indices(indices.shape)[0], indices], 1), 1
        )
        # TODO: optimise this somehow
        indices = np.array(
            [
                [embeddings[col_index].tolist().index(v) for v in row]
                for col_index, row in enumerate(values.tolist())
            ]
        )
        return values, indices

    def community_detection_cpu(self, embeddings):
        self.init_max_size = min(self.init_max_size, len(embeddings))
        cos_scores = self.cosine(embeddings)
        top_k_values, _ = self.topk(cos_scores, k=self.min_community_size)

        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= self.threshold:
                new_cluster = []

                top_val_large, top_idx_large = self.topk(
                    cos_scores[i], k=self.init_max_size
                )
                top_idx_large = top_idx_large.flatten().tolist()
                top_val_large = top_val_large.flatten().tolist()

                if top_val_large[-1] < self.threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < self.threshold:
                            break

                        new_cluster.append(idx)
                else:
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= self.threshold:
                            new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        extracted_communities = sorted(
            extracted_communities, key=lambda x: len(x), reverse=True
        )

        unique_communities = []
        extracted_ids = set()

        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break

            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)

        print(f"There were {len(unique_communities)} communities found.")
        return unique_communities

    def community_detection_gpu(self, embeddings):
        try:
            from sentence_transformers.util import community_detection
        except ModuleNotFoundError:
            raise MissingPackageError("sentence-transformers")
        return community_detection(
            embeddings,
            self.threshold,
            self.min_community_size,
            self.init_max_size,
        )
