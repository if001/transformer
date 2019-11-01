from keras.layers import Input, Lambda, Dense
from keras.layers import Softmax
from keras.models import Model
from keras import backend as K


class SimpleAttention():
    def __init__(self, q_length, m_length, depth):
        self.input = Input(shape=(q_length, depth))
        self.memory = Input(shape=(m_length, depth))
        self.attention_mask = Input(shape=(1, q_length, m_length))
        self.depth = depth

        self.model = Model([self.input, self.memory], self.call())

    def call(self):
        q = Dense(self.depth, use_bias=False, name="q_dense_layer")(self.input)
        k = Dense(self.depth, use_bias=False,
                  name="k_dense_layer")(self.memory)
        v = Dense(self.depth, use_bias=False,
                  name="v_dense_layer")(self.memory)

        logit = self.__dot([q, self.__transpose(k, name="t")], name="logit")

        scaled_logit = self.__scale(logit, name="scale")
        attention_weight = self.__softmax_layer(name="softmax")(scaled_logit)

        attention_output = self.__dot([attention_weight, v], name="attention")
        return Dense(self.depth, use_bias=False, name="output_dense_layer")(attention_output)

    def __dot(self, x, name=None):
        return Lambda(lambda x: K.batch_dot(x[0], x[1]), name=name)(x)

    def __transpose(self, x, name=None):
        # without batch
        return Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name=name)(x)

    def __softmax_layer(self, name=None):
        return Lambda(lambda x: K.softmax(x), name=name)

    def __scale(self, x, name=None):
        return Lambda(lambda x: x/(self.depth**-0.5), name=name)(x)


def main():
    s = SimpleAttention(1, 2, 3)
    # s.model.summary()
    s.model.summary()

    inp = [[1, 2, 3]]
    msk = [[False], [False], [True]]

    a = Lambda(masked_func, name=name)([inp, msk])
    r = Lambda(lambda x: K.softmax(x), name=name)(a)


def masked_func(x, msk):
    replace =


if __name__ == "__main__":
    main()
