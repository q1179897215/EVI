from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

multi_embedding_vocabulary_size = {
    "101": 238635,
    "121": 98,
    "122": 14,
    "124": 3,
    "125": 8,
    "126": 4,
    "127": 4,
    "128": 3,
    "129": 5,
    "205": 467298,
    "206": 6929,
    "207": 263942,
    "216": 106399,
    "508": 5888,
    "509": 104830,
    "702": 51878,
    "853": 37148,
    "301": 4,
}
using_feature_ids = [str(i) for i in range(330)]
using_feature_ids.pop(295)
in_feature_names = [i for i in range(0, len(using_feature_ids))]
cpp_feature_names = sorted(list(multi_embedding_vocabulary_size.keys()))
cpp_embedding_vocabulary_size = 737946
in_embedding_vocabulary_size = 10614790


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims,
        dropout,
        drop_last_dropout=False,
        output_layer=True,
    ):
        super().__init__()
        layers = list()
        for i, embed_dim in enumerate(embed_dims):
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout[i] > 0:
                layers.append(torch.nn.Dropout(p=dropout[i]))
            input_dim = embed_dim
        if drop_last_dropout == True:
            layers.pop()
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class AlldataEmbeddingLayer(torch.nn.Module):
    def __init__(self, batch_type="ccp", embedding_size=5):
        super().__init__()
        self.batch_type = batch_type
        self.numerical_num = 63

        if batch_type == "ccp":
            self.feature_names = cpp_feature_names
            self.embedding_layer = torch.nn.Embedding(737946, embedding_size)
            self.embed_output_dim = 18 * embedding_size
        elif batch_type == "in":
            self.feature_names = in_feature_names
            self.embedding_layer = torch.nn.Embedding(10614790, embedding_size)
            self.embed_output_dim = len(self.feature_names) * embedding_size
        elif batch_type == "fr":
            self.field_dims = [9, 4, 7, 2, 20, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "nl":
            self.field_dims = [9, 4, 7, 2, 20, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "es":
            self.field_dims = [8, 4, 7, 2, 19, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size
        elif batch_type == "us":
            self.field_dims = [10, 4, 7, 2, 21, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
            self.embedding_layer = torch.nn.Embedding(sum(self.field_dims), embedding_size)
            self.numerical_layer = torch.nn.Linear(self.numerical_num, embedding_size)
            self.embed_output_dim = (len(self.field_dims) + 1) * embedding_size

        # self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight.data)

    def forward(self, x):
        if self.batch_type == "ccp":
            feature_embedding = []
            for name in self.feature_names:
                embed = self.embedding_layer(x[name])
                # print(embed.shape) --> [batch_size, embedding_size] = [2000, 5]
                feature_embedding.append(embed)
            return torch.cat(feature_embedding, 1)
        elif self.batch_type == "in":
            feature_embedding = []
            for name in range(len(self.feature_names)):
                embed = self.embedding_layer(x[:, name, :])
                embed_sum = torch.sum(embed, dim=1)
                # print(embed.shape) --> [batch_size, embedding_size] = [2000, 5]
                feature_embedding.append(embed_sum)
            return torch.cat(feature_embedding, 1)
        else:
            categorical_x, numerical_x = x
            categorical_emb = self.embedding_layer(categorical_x)
            numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
            return torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)

    def get_embed_output_dim(self):
        return self.embed_output_dim


class BatchTransform:
    def __init__(self, batch_type="ccp"):
        self.batch_type = batch_type
        self.feature_num = len(using_feature_ids)
        self.single_feature_len = 3

    def __call__(self, batch):
        if self.batch_type == "in":
            click, conversion, features = (
                batch["click"].squeeze(1).float(),
                batch["conversion"].squeeze(1).float(),
                batch["features"],
            )
            features = features.reshape(len(features), self.feature_num, self.single_feature_len)

        elif self.batch_type == "ccp":
            click, conversion, features = batch
            click = click.float()
            conversion = conversion.float()
        else:
            click, conversion, features = batch

        return click, conversion, features


if __name__ == "__main__":
    embed = EmbeddingLayer(100, 5)
    data = torch.randint(0, 100, (2000, 90))
