# ARG BASE_IMAGE=pytorch/pytorch:latest
ARG BASE_IMAGE=pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

FROM ${BASE_IMAGE}

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH /workspace

ENTRYPOINT /bin/bash
