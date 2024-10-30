import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class Model:
    def __init__(self, model_path, tokenizer_path):
        self.model=tf.keras.models.load_model(model_path)
        with open(tokenizer_path) as json_file:
            tokenizer_json = json_file.read()
        # Reconstruir el tokenizer desde el archivo JSON
        self.tokenizer = tokenizer_from_json(tokenizer_json)
    def decode(self, predictions, greedy):
    predicted_classes = tf.argmax(predictions, axis=-1)
    mask = tf.not_equal(predicted_classes, 0)
    seq_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
    predictions = tf.cast(predictions, dtype=tf.float32)
    predictions=tf.transpose(predictions,[1,0,2])

    if greedy:
        decoded, _ = tf.nn.ctc_greedy_decoder(inputs=predictions, sequence_length=seq_lengths)

    else:
        decoded, _ = tf.nn.ctc_beam_search_decoder(inputs=predictions, sequence_length=seq_lengths)
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()

    # Extraer las secuencias decodificadas eliminando los -1 (valores de relleno)
    decoded_sequences = []
    for seq in decoded_dense:
        decoded_sequences.append([val for val in seq if val != -1])

    return decoded_sequences
    def predict(self, points):
        predictions=self.model.predict(points)
        decoded_sequences=self.decode(predictions, greedy=False)
        prediction=self.tokenizer.sequence_to_text(decoded_sequences)
        return prediction