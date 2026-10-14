# Pinned, not :latest: the server version decides which API features exist
# (`think`, JSON-schema `format`, both needed here -- 0.9 and later) and how a
# model is run, so a rebuild must not change it silently. Picked on 2026-10-15
# as the newest stable (non-rc) release more than four weeks old; bump it on
# purpose and re-run the eval. The model weights are a separate matter: see
# "Reproducibility" in docs/infrastructure.md.
FROM ollama/ollama:0.34.2

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
