"""Prompt and portable package identifiers."""

PROMPT_V1 = "jevora-v0.1-choice"
PROMPT_V2 = "jevora-v0.2-lm-head-codes"
PROMPT_NATIVE_CHAT = "jevora-v1.5-native-chat-codes-v1"
CHECKPOINT_FORMAT = "jevora-checkpoint-v2"
RELEASE_FORMAT = "jevora-release-v1"
SHARDED_RELEASE_FORMAT = "jevora-release-v2"
SAFETENSORS_WEIGHT_FORMAT = "safetensors-sharded-v1"
SAFETENSORS_INDEX = "model.safetensors.index.json"


class InvalidInput(ValueError):
    """The supplied decision request is invalid."""


class InputCapacityError(InvalidInput):
    """The request exceeds the supported option or token limits."""
