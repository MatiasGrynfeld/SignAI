import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class Model:
    def __init__(self, model_path, tokenizer_path):
        self.model=tf.keras.models.load_model(model_path)
        self.decoder=self.model.get_layer('decoder')
        self.encoder=self.model.get_layer('encoder')
        self.cnn=self.model.get_layer('cnn')
        with open(tokenizer_path) as json_file:
            tokenizer_json = json_file.read()
        # Reconstruir el tokenizer desde el archivo JSON
        self.tokenizer = tokenizer_from_json(tokenizer_json)
        self.start_token_id=self.tokenizer.word_index['<start>']
        self.end_token_id=self.tokenizer.word_index['<end>']
    
    def predict(self, points):
        max_len=400
        decoder_input = tf.constant([[self.start_token_id]])
        output_sequence = []
        points=tf.expand_dims(tf.expand_dims(points,axis=0), axis=-1)


        cnn_output=self.cnn(points, training=False)
        print(cnn_output.shape)


        encoder_outputs, encoder_states = self.encoder(cnn_output, training=False)
        states= encoder_states


        for _ in range(max_len):
            decoder_outputs, state_h, state_c =self.decoder([decoder_input, encoder_outputs, states], training=False)
            decoder_outputs=tf.cast(tf.argmax(decoder_outputs, axis=-1), dtype=tf.int32)
            output_sequence.append(int(decoder_outputs))
            states=[state_h, state_c]
            decoder_input = tf.expand_dims(tf.expand_dims(decoder_outputs[0,-1],axis=0),axis=0)
            if int(decoder_outputs[0,-1]) == self.end_token_id or int(decoder_outputs[0,-1])==0:
                break

        translation = self.tokenizer.sequences_to_texts([output_sequence])
        return translation