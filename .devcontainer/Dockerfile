FROM ros:jazzy
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies upfront
RUN python3 -m pip install --break-system-packages \
    numpy scipy scikit-learn scikit-image open3d pyyaml

ENV SHELL=/bin/bash

USER $USERNAME
CMD ["/bin/bash"]
