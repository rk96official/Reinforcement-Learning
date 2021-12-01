
from deepLearning.nlu.rasa import  rasa


def set_nlu(params, nlu_type, top_intents, **kargs):
        if nlu_type == 'rasa':
            nlu = rasa(params, top_intents, **kargs)
        return nlu
