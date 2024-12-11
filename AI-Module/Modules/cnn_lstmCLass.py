import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, TimeDistributed, Attention, Conv1D, Concatenate, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.mixed_precision import set_global_policy
import json

# Configurar la política de precisión mixta
set_global_policy('mixed_float16')
@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        pass
    def call(self, inputs):
        max_len = tf.shape(inputs)[1]  # Longitud máxima de la secuencia
        batch_size = tf.shape(inputs)[0]

        # Calcula las posiciones y términos divisores
        position = tf.cast(tf.range(max_len), tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))

        # Calcula las codificaciones posicionales
        pos_sin = tf.math.sin(position * div_term)
        pos_cos = tf.math.cos(position * div_term)

        # Combina sinusoides en la dimensión de características
        pos_encoding = tf.concat([pos_sin, pos_cos], axis=-1)

        # Ajusta las dimensiones para que coincidan con el batch
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Añade dimensión de batch
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])  # Repite para cada ejemplo del batch

        return inputs + tf.cast(pos_encoding, inputs.dtype)
    def get_config(self):
        # Permite guardar los parámetros de inicialización
        config = super(PositionalEncoding, self).get_config()
        config.update({"d_model": self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        # Permite cargar la capa desde una configuración
        return cls(**config)
@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self,num_heads=4, key_dim=128, ffn_units=256, dropout_rate=0.2,ln_rate=1e-6, **kwargs):
        super(TransformerBlock, self).__init__( **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.ln_rate=ln_rate

        self.attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ffn_units, activation="relu"),
            Dense(key_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=ln_rate)
        self.layernorm2 = LayerNormalization(epsilon=ln_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    def build(self, input_shape):
        pass
    def call(self, inputs, training=None):
        # tf.print(tf.shape(inputs))
        attn_output = self.attention_layer(query=inputs, value=inputs,key=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
            "ln_rate": self.ln_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@register_keras_serializable()
class DecoderTransformerBlock(layers.Layer):
    def __init__(self,num_heads=4, key_dim=128, ffn_units=512, dropout_rate=0.2,ln_rate=1e-6, **kwargs):
        super(DecoderTransformerBlock, self).__init__( **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.ln_rate=ln_rate

        self.attention_layer_1 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attention_layer_2 = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

        self.ffn = tf.keras.Sequential([
            Dense(ffn_units, activation="relu"),
            Dense(key_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=ln_rate)
        self.layernorm2 = LayerNormalization(epsilon=ln_rate)
        self.layernorm3 = LayerNormalization(epsilon=ln_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, inputs, encoder_outputs, cache, training=None):

        input_secuencia = tf.concat([cache, inputs], axis=1)
        causal_mask = tf.linalg.band_part(
            tf.ones((tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(input_secuencia)[1])), -1, 0)

        attn_output = self.attention_layer_1(query=inputs, value=input_secuencia, key=input_secuencia, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        attn_output = self.attention_layer_2(query=out1, value=encoder_outputs, key=encoder_outputs)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = self.layernorm2(out1 + attn_output)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output= self.layernorm3(out2 + ffn_output)

        cache = input_secuencia
        return output, cache

    def get_config(self):
        config = super(DecoderTransformerBlock, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ffn_units": self.ffn_units,
            "dropout_rate": self.dropout_rate,
            "ln_rate": self.ln_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@register_keras_serializable()
class GlobalTokens(layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalTokens, self).__init__(**kwargs)
        self.embedding_layer=Embedding(input_dim=5, output_dim=128)

    def build(self, input_shape):
        pass
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        token_indices = tf.range(5, dtype=tf.int64)
        token_indices = tf.tile(tf.expand_dims(token_indices, axis=0), [batch_size, 1])
        token_embeddings = self.embedding_layer(token_indices)
        return tf.concat((token_embeddings,inputs), axis=1)

    def get_config(self):
        config = super(GlobalTokens, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Model:
    def __init__(self, model_path, tokenizer_path):
        self.model=tf.keras.models.load_model(model_path, safe_mode=False)
        self.decoder=self.model.get_layer('decoder')
        self.encoder=self.model.get_layer('encoder')
        self.tokenizer=self.create_tokenizer(tokenizer_path)
        self.start_token_id=self.tokenizer.word_index['<start>']
        self.end_token_id=self.tokenizer.word_index['<end>']
    
    def create_tokenizer(self, tokenizer_path):
        with open(tokenizer_path) as json_file:
          tokenizer_json = json.load(json_file)

        # Reconstruir el tokenizer desde el archivo JSON
        tokenizer = tokenizer_from_json(tokenizer_json)
        start_token='<start>'
        end_token='<end>'
        padding_token='<pad>'
        tokenizer.word_index[start_token]=len(tokenizer.word_index)+1
        tokenizer.word_index[end_token]=len(tokenizer.word_index)+1
        tokenizer.word_index[padding_token]=0
        tokenizer.word_index[start_token], tokenizer.word_index[end_token], tokenizer.word_index[padding_token]
        tokenizer.index_word[0]=padding_token
        tokenizer.index_word[tokenizer.word_index[start_token]]=start_token
        tokenizer.index_word[tokenizer.word_index[end_token]]=end_token
        tokenizer.index_word[0], tokenizer.index_word[tokenizer.word_index[start_token]], tokenizer.index_word[tokenizer.word_index[end_token]]
        print(tokenizer.word_index[end_token])
        return tokenizer

    def predict(self, points):
        max_len=100
        points=tf.cast(points, tf.float16)
        labels = tf.concat([
            tf.zeros((21,), dtype=tf.int32),    # Mano 1 -> etiqueta 0
            tf.ones((21,), dtype=tf.int32),     # Mano 2 -> etiqueta 1
            tf.fill((33,), 2),         # Pose -> etiqueta 2
            tf.fill((543 - 21 - 21 - 33,), 3)   # Cara -> etiqueta 3
        ], axis=0)
        points=tf.reshape(points, (tf.shape(points)[0], 543,4))
        # Expandir las etiquetas para que coincidan con (batch, time, 543, 1)
        labels = tf.reshape(labels, (1, 543, 1))
        labels = tf.tile(labels, (tf.shape(points)[0], 1,1))

        num_classes = 4  # Número de categorías
        labels_one_hot = tf.one_hot(labels, depth=num_classes, dtype=tf.float16)
        labels_one_hot = tf.squeeze(labels_one_hot, axis=-2)

        # Concatenar las etiquetas con la entrada reestructurada
        points = tf.concat([points, tf.cast(labels_one_hot, tf.float16)], axis=-1)
        points=tf.expand_dims(points,axis=0)
        encoder_outputs = self.encoder(points, training=False)
        
        decoder_outputs=tf.zeros((1,0,16325), dtype=tf.float16)
        cache=tf.zeros((1, 0, 256), dtype=tf.float16)
        predicted_token=tf.expand_dims(tf.cast(self.tokenizer.word_index['<start>'], dtype=tf.int32), axis=0)
        print(predicted_token)
        decoder_input=tf.expand_dims(predicted_token,axis=0)

        # Execute the while loop
        for t in range(max_len):
            decoder_input.set_shape([1,None])
            decoder_output, cache1 = self.decoder(
                [decoder_input, encoder_outputs, cache],
                training=False
            )
            decoder_outputs = decoder_output
            decoder_input=tf.cast(tf.argmax(decoder_output, axis=-1),dtype=tf.int32)
            decoder_input=tf.concat([tf.expand_dims(predicted_token, axis=0), decoder_input], axis=1)
            if decoder_input[0,-1].numpy()==self.end_token_id:
                break
        outputs=tf.argmax(decoder_outputs, axis=-1)
        translation=self.tokenizer.sequences_to_texts(outputs.numpy())
        return translation