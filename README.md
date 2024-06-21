# BatchByTokens
A simple wrapper for the HuggingFace Seq2SeqTrainer that allows for batching by tokens.

## Installation
``` python setup.py install ```

## Usage
``` from batchbytokens import BBTTrainer ```

For passing in the batch size, just pass in to Seq2SeqTrainingArguments as you normally would.
For now, the eval batch size is tied to the training batch size. 
