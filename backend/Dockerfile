FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch must be installed BEFORE requirements.txt, and from the CUDA 12.1 index.
#
# sentence-transformers pulls torch transitively, and PyPI's default wheel is
# now a CUDA 13 build whose compiled architectures start at sm_75 (Turing).
# The target GPU is a GTX 1080 Ti -- Pascal, sm_61 -- which CUDA 13 dropped, so
# the default wheel sees the card but has no kernels for it and the reranker
# silently falls back to CPU. cu121 is the newest index still shipping sm_61.
#
# Installing it first means the requirements.txt resolve finds torch already
# satisfied and leaves it alone. Check with torch.cuda.get_arch_list() after
# changing this: 'sm_61' must be in the list.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "recommender.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
