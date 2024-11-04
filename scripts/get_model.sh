#!/bin/sh

for f in model.safetensors tokenizer.json config.json; do
	wget https://huggingface.co/TaylorAI/bge-micro-v2/resolve/main/$f
done

