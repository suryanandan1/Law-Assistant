import os

from dotenv import load_dotenv
from loguru import logger
# Mistral SDK 2.x moved the client from the package root to
# ``mistralai.client``.  Keep the fallback so this project also works with
# the 1.x SDK specified originally.
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


load_dotenv()


class MistralService:

    def __init__(self):

        api_key = os.getenv(
            "MISTRAL_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "MISTRAL_API_KEY not found"
            )

        self.client = Mistral(
            api_key=api_key
        )

        self.model = os.getenv(
            "MISTRAL_MODEL",
            "mistral-large-latest"
        )

    def generate_answer(
        self,
        prompt: str
    ) -> str:

        try:

            response = (
                self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            logger.exception(e)

            raise
