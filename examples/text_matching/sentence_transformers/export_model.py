# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import paddle.nn as nn

from paddlenlp.transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, default="ernie-1.0", help="The path to model parameters to be loaded.")
parser.add_argument(
    "--output_path", type=str, default="./export", help="The path of model parameter in static graph to be saved."
)
args = parser.parse_args()


class SentenceTransformer(nn.Layer):
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"] * 3, 2)

    def forward(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
    ):
        query_token_embedding, _ = self.ptm(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )
        query_token_embedding = self.dropout(query_token_embedding)
        query_attention_mask = paddle.unsqueeze(
            (query_input_ids != self.ptm.pad_token_id).astype(self.ptm.pooler.dense.weight.dtype), axis=2
        )
        # Set token embeddings to 0 for padding tokens
        query_token_embedding = query_token_embedding * query_attention_mask
        query_sum_embedding = paddle.sum(query_token_embedding, axis=1)
        query_sum_mask = paddle.sum(query_attention_mask, axis=1)
        query_mean = query_sum_embedding / query_sum_mask

        title_token_embedding, _ = self.ptm(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask
        )
        title_token_embedding = self.dropout(title_token_embedding)
        title_attention_mask = paddle.unsqueeze(
            (title_input_ids != self.ptm.pad_token_id).astype(self.ptm.pooler.dense.weight.dtype), axis=2
        )
        # Set token embeddings to 0 for padding tokens
        title_token_embedding = title_token_embedding * title_attention_mask
        title_sum_embedding = paddle.sum(title_token_embedding, axis=1)
        title_sum_mask = paddle.sum(title_attention_mask, axis=1)
        title_mean = title_sum_embedding / title_sum_mask

        sub = paddle.abs(paddle.subtract(query_mean, title_mean))
        projection = paddle.concat([query_mean, title_mean, sub], axis=-1)

        logits = self.classifier(projection)

        return logits


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    pretrained_model = AutoModel.from_pretrained(args.params_path)

    model = SentenceTransformer(pretrained_model)
    model.eval()

    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="query_input_ids"),
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="title_input_ids"),
    ]
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(model, input_spec=input_spec)

    # Save in static graph model.
    save_path = os.path.join(args.output_path, "float32")
    paddle.jit.save(model, save_path)
