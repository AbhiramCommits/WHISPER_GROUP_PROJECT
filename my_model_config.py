from whisper.model import ModelDimensions, Whisper

# SMALL_DIMS = ModelDimensions(
#     n_mels=80,
#     n_audio_ctx=1500,
#     n_audio_state=512,   # up from 384
#     n_audio_head=8,      # up from 6
#     n_audio_layer=6,     # up from 4
#     n_vocab=51865,
#     n_text_ctx=448,
#     n_text_state=512,
#     n_text_head=8,
#     n_text_layer=6,
# )


SMALL_DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)

def get_model():
    return Whisper(SMALL_DIMS)