# Requires the refreshed current-source intermediate image recorded in README.md.
FROM scalarforensic:internal-nvidia-20260912-refreshed@sha256:d0c73b726fcee8d3494550422d0f57e17a2973288a636071ba7982c68ce05596
USER root
COPY pip-26.2.1-py3-none-any.whl requirements-pip.txt /tmp/pip-update/
RUN python -m pip install --no-cache-dir --no-index --no-deps --require-hashes \
      --find-links /tmp/pip-update -r /tmp/pip-update/requirements-pip.txt \
    && python -m pip check \
    && rm /tmp/pip-update/pip-26.2.1-py3-none-any.whl /tmp/pip-update/requirements-pip.txt
USER 10001:10001
