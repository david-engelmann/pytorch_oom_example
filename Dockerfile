# [Choice] Ubuntu version (use hirsute or bionic on local arm64/Apple Silicon): hirsute, focal, bionic, jammy
ARG VARIANT="dev-hirsute"
FROM mcr.microsoft.com/vscode/devcontainers/base:${VARIANT} as ubuntu_base
ENV VARIANT $VARIANT

# [Choice] Virtual Enviroment Path
ENV VIRTUAL_ENV="/opt/venv"
ARG VIRTUAL_ENV $VIRTUAL_ENV

# Section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
	&& apt-get -y install --no-install-recommends \
		build-essential \
		libpython3-dev \
		libboost-all-dev \
		python3 \
		python3-pip \
		python3-wheel \
		#python${PYTHON_VERSION}-dev \
		#python${PYTHON_VERSION}-venv \
		python3-setuptools \
		python3-dev \
		python3-venv \
	&& apt-get update && apt-get -y upgrade

WORKDIR /workspaces/pytorch_oom_example

# Make venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Activate venv
RUN . $VIRTUAL_ENV/bin/activate

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools wheel \
	&& pip3 install --no-cache-dir -r requirements.txt

RUN alias python=python3

CMD ["/bin/bash"]
