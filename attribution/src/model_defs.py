import tensorflow as tf
from tensorflow.keras import layers, Model

def gru_att_cls(seq_len:int, feature_len:int,
                binary_idx:list, category_idx:list,
                category_emb_size:list, emb_dim:int,
                gru_seq_len: int, gru_units:int, dropout_rate:float, att_units:int):

    def debug_shape(name):
        # user ex: query = debug_shape("query")(query)
        return tf.keras.layers.Lambda(
            lambda t: (tf.print(f"{name}:", tf.shape(t)), t)[1],
            name=f"dbg_{name}"
        )

    # 1) Input: (B, T, F)
    inputs = tf.keras.Input(shape=(seq_len, feature_len), dtype=tf.float32, name="inputs")

    # 2) binary / category 분리
    # - binary: float32 유지 (패딩은 -1.0)
    # - category: 임베딩 위해 int32로 캐스팅 (패딩은 0)
    bi_inputs = tf.keras.layers.Lambda(
        lambda t: tf.gather(
            t,
            indices=tf.constant(binary_idx, dtype=tf.int32),
            axis=-1
        ),
        name="pick_binary"
    )(inputs)
    cat_inputs_f = tf.keras.layers.Lambda(
        lambda t: tf.gather(
            t,
            indices=tf.constant(category_idx, dtype=tf.int32),
            axis=-1
        ),
        name="pick_cats_float"
    )(inputs)
    cat_inputs = tf.keras.layers.Lambda(
        lambda t: tf.cast(t, tf.int32),
        name="cast_cats_to_int")(cat_inputs_f)  # (B,T,K) int32

    # 3) 타임스텝 마스크 (cat + binary를 함께 고려. 둘 중 하나라도 유효하면 True)
    # - cat 중 하나라도 0이 아니면 유효
    # - binary 중 하나라도 -1.0이 아니면 유효
    valid_mask = tf.keras.layers.Lambda(
        lambda xs: tf.logical_or(
            tf.reduce_any(tf.not_equal(xs[0], 0), axis=-1),  # (B,T)
            tf.reduce_any(tf.not_equal(xs[1], -1.0), axis=-1)  # (B,T)
        ), name="valid_mask_cat_and_bin")([cat_inputs, bi_inputs])  # (B,T) bool

    # 4) 임베딩 (카테고리 K개 → 각자 임베딩 후 concat)
    embeds_cat = []
    for j, vocab in enumerate(category_emb_size):
        e = tf.keras.layers.Embedding(
            input_dim=vocab + 1,  # +1은 패딩토큰(0)을 위한 슬롯
            output_dim= emb_dim,  # 임베딩 차원 (원하면 조정)
            mask_zero=False,  # True(임베딩 내부 마스크 생성 (직접 만든 마스크 사용))
            name=f"emb_cat{j}"
        )(cat_inputs[..., j])  # (B,T,16) # j번째 카테고리 피처
        embeds_cat.append(e)
    x_cat = tf.keras.layers.Concatenate(axis=-1, name="concat_cat_embs")(embeds_cat)  # (B,T,16*K)

    # 5) 바이너리 + 카테고리 결합
    x = tf.keras.layers.Concatenate(axis=-1, name="concat_cat_bin")([bi_inputs, x_cat])  # (B,T, Cb + 16K)

    # 6) GRU
    x_seq = x  # x_seq: (B, T, H)
    for i in range(gru_seq_len):
        x_seq = tf.keras.layers.GRU(gru_units,
                                    return_sequences=True,
                                    dropout=dropout_rate, recurrent_dropout=0.0,
                                    name=f"gru_{i}")(x_seq, mask = valid_mask)

    # 7) Attention
    # cls query
    query = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="gap_query_init")(x_seq)  # (B,1,H)
    query = tf.keras.layers.Dense(att_units, activation='tanh', name="learned_query")(query)  # (B,1,H)
    # attention mask
    attn_mask = tf.keras.layers.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.bool), 1))(valid_mask)
    # attention
    mha = tf.keras.layers.MultiHeadAttention(
        num_heads=1, key_dim=att_units, #output_shape=gru_units,
        dropout=0.0, name="mha_time"
    )
    context, scores = mha(
        query=query,  # (B,1,H)
        value=x_seq,  # (B,T,H)
        key=x_seq,  # (B,T,H)
        attention_mask=attn_mask, # (B,1,T)
        return_attention_scores=True
    )  # context: (B,1,H), scores: (B,1,T)
    # reshape
    # context: (B, T_q, H) = (B,1,H) -> (B,H)
    # scores: (B, T_q, T) = (B,1,T)  -> (B,T)
    context = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1),name="ctx_drop_time")(context)
    alpha = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1),name="alpha_drop_query")(scores)

    # 8) 분류
    out = tf.keras.layers.Dense(1, activation="sigmoid")(context)

    # 학습용 / 가중치 확인용 모델
    train_model = tf.keras.Model(inputs, out, name="gru_builtin_attn")
    attn_model  = tf.keras.Model(inputs, [out, alpha], name="gru_builtin_attn_with_alpha")

    return train_model, attn_model


def transformer_att_cls(seq_len:int, feature_len:int,
                binary_idx:list, category_idx:list,
                category_emb_size:list, emb_dim:int,
                num_heads:int, key_dim:int, dense_dim:int, encoder_n:int,
                dropout_rate:float, att_units:int):
    def debug_shape(name):
        # user ex: query = debug_shape("query")(query)
        return tf.keras.layers.Lambda(
            lambda t: (tf.print(f"{name}:", tf.shape(t)), t)[1],
            name=f"dbg_{name}"
        )

    # 1) Input: (B, T, F)
    inputs = tf.keras.Input(shape=(seq_len, feature_len), dtype=tf.float32, name="inputs")
    # 2) binary / category 분리
    # - binary: float32 유지 (패딩은 -1.0)
    # - category: 임베딩 위해 int32로 캐스팅 (패딩은 0)
    bi_inputs = tf.keras.layers.Lambda(
        lambda t: tf.gather(
            t,
            indices=tf.constant(binary_idx, dtype=tf.int32),
            axis=-1
        ),
        name="pick_binary"
    )(inputs)
    cat_inputs_f = tf.keras.layers.Lambda(
        lambda t: tf.gather(
            t,
            indices=tf.constant(category_idx, dtype=tf.int32),
            axis=-1
        ),
        name="pick_cats_float"
    )(inputs)
    cat_inputs = tf.keras.layers.Lambda(
        lambda t: tf.cast(t, tf.int32),
        name="cast_cats_to_int")(cat_inputs_f)  # (B,T,K) int32

    # 3) 임베딩 (카테고리 K개 → 각자 임베딩 후 concat)
    embeds_cat = []
    for j, vocab in enumerate(category_emb_size):
        e = tf.keras.layers.Embedding(
            input_dim=vocab + 1,  # +1은 패딩토큰(0)을 위한 슬롯
            output_dim=emb_dim,  # 임베딩 차원 (원하면 조정)
            mask_zero=False,  # True(임베딩 내부 마스크 생성 (직접 만든 마스크 사용))
            name=f"emb_cat{j}"
        )(cat_inputs[..., j])  # (B,T,16) # j번째 카테고리 피처
        embeds_cat.append(e)
    x_cat = tf.keras.layers.Concatenate(axis=-1, name="concat_cat_embs")(embeds_cat)  # (B,T,16*K)
    # 4) 바이너리 + 카테고리 결합
    x = tf.keras.layers.Concatenate(axis=-1, name="concat_cat_bin")([bi_inputs, x_cat])  # (B,T, Cb + 16K)
    # 5) Dense layer (F --> D)
    x = layers.Dense(dense_dim, name="proj_to_d")(x)  # (B,T,D)

    # 6) Positional Embedding
    pos_ids = tf.range(0, seq_len)  # (T,)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=dense_dim, name="pos_emb")(pos_ids)  # (T,D)
    x = x + tf.expand_dims(pos_emb, axis=0)  # (B,T,D) (position emb는 bias처럼 동작 --> +)
    x = layers.Dropout(dropout_rate)(x)

    # 7) encoder의 Self-Attention에 들어갈 마스크 정의 (패딩 & 단방향(causal))
    valid_mask = tf.keras.layers.Lambda(
        lambda xs: tf.logical_or(
            tf.reduce_any(tf.not_equal(xs[0], 0), axis=-1),  # (B,T)
            tf.reduce_any(tf.not_equal(xs[1], -1.0), axis=-1)  # (B,T)
        ), name="valid_mask_cat_and_bin")([cat_inputs, bi_inputs])  # (B,T) bool
    # 패딩 마스크
    q_keep = layers.Lambda(lambda m: tf.expand_dims(m, 2), name="q_keep")(valid_mask)  # (B,T,1)
    k_keep = layers.Lambda(lambda m: tf.expand_dims(m, 1), name="k_keep")(valid_mask)  # (B,1,T)
    pad_mask = layers.Lambda(
        lambda tensors: tf.logical_and(
            tf.cast(tensors[0], tf.bool),
            tf.cast(tensors[1], tf.bool)
        ),
        name="pad_mask",
        output_shape=lambda shapes: (
            shapes[0][0],  # batch dim (None)
            shapes[0][1],  # T from q_keep:  (None, T, 1)
            shapes[1][2],  # T from k_keep:  (None, 1, T)
        )
    )([q_keep, k_keep])  # (B,T,T)
    # causal 마스크 : 하삼각 True → (1,T,T)
    causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0)
    causal = tf.expand_dims(causal, 0)  # (1,T,T)
    # 최종 마스크
    enc_attn_mask = layers.Lambda(
        lambda m: tf.logical_and(
            tf.cast(m, tf.bool),
            causal  # 상수라 그냥 사용 가능
        ),
        name="enc_attn_mask",
        output_shape=lambda input_shape: (
            input_shape[0],  # batch dim (None)
            input_shape[1],  # T
            input_shape[2],  # T
        )
    )(pad_mask)  # (B,T,T)

    # 8) Encoder
    def enc_block(z, name):
        # Self-Attention
        attn_layer = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name=f"{name}_mha"
        )
        attn_out, _ = attn_layer(
            z, z,
            attention_mask=enc_attn_mask,
            return_attention_scores=True
        )  # (B,T,D)
        z = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(
            z + layers.Dropout(dropout_rate)(attn_out)  # Residual
        )
        # FFN
        ff = layers.Dense(4 * dense_dim, activation="relu", name=f"{name}_ff1")(z)
        ff = layers.Dense(dense_dim, name=f"{name}_ff2")(ff)
        z = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(
            z + layers.Dropout(dropout_rate)(ff)
        )
        return z

    z = x
    for i in range(encoder_n):
        z = enc_block(z, name=f"enc{i + 1}")

    # 9) cls 토큰: Label-level Attention (나중 쿼리)
    # GRU 버전처럼: GAP로 초기 쿼리 만들고 작은 MLP 한 번
    q = tf.keras.layers.GlobalAveragePooling1D(keepdims=True, name="query_init_avg")(z)  # (B,1,D)
    q = tf.keras.layers.Dense(att_units, activation="tanh", name="query_dense")(q)  # (B,1,att_units)
    # attention mask
    attn_mask = tf.keras.layers.Lambda(lambda m: tf.expand_dims(tf.cast(m, tf.bool), 1))(valid_mask)
    # attention
    att = tf.keras.layers.MultiHeadAttention(
        num_heads=1, key_dim=att_units, name="mha_time"
    )
    context, scores = att(
        query=q,
        value=z,
        key=z,
        attention_mask=attn_mask,
        return_attention_scores=True
    )
    context = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1), name="ctx_drop_time")(context)
    alpha = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1), name="alpha_drop_query")(scores)

    # 11) 분류
    out = tf.keras.layers.Dense(1, activation="sigmoid")(context)

    # 학습용 / 가중치 확인용 모델
    train_model = tf.keras.Model(inputs, out, name="trf_label_attn")
    attn_model = tf.keras.Model(inputs, [out, alpha], name="trf_label_attn_with_alpha")

    return train_model, attn_model