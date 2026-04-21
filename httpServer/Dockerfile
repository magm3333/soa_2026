FROM python:3.9.18-slim-bullseye
LABEL maintainer="Magm magm@iua.edu.ar"


WORKDIR /www

ARG UID=1000
ARG GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g "${GID}" magm \
    && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" magm \
    && chown magm:magm -R /www

USER magm

ENV PYTHONUNBUFFERED="true" \
    PYTHONPATH="." \
    PATH="${PATH}:/home/magm/.local/bin" \
    USER="magm" \
    MAGM_PORT="8080"


COPY --chown=magm:magm httpServer /usr/local/bin
COPY --chown=magm:magm index.html .

RUN chmod +x /usr/local/bin/httpServer

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=2s --retries=3 --start-interval=5s  CMD curl -fsS http://localhost:$MAGM_PORT/health || exit 1

CMD ["sh", "-c", "httpServer 0.0.0.0:$MAGM_PORT /www"]  




